from datetime import datetime
import os
import io
import tempfile

from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from json_repair import loads as json_repair_loads
from pydantic import BaseModel, Field
from pypdf import PdfReader

import google.generativeai as genai
from ollama import Client as OllamaClient

from docling.document_converter import DocumentConverter


load_dotenv()

# --- LLM Client Configuration (Conditional based on LLM_PROVIDER) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

GEMINI_MODEL_NAME = "gemini-1.5-flash"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1") 

gemini_model = None
ollama_client = None

if LLM_PROVIDER == "google":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set for 'google' LLM_PROVIDER.")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Backend: LLM Provider: Google Gemini ({GEMINI_MODEL_NAME})")
elif LLM_PROVIDER == "ollama":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        ollama_client = OllamaClient(host=OLLAMA_BASE_URL)
        ollama_client.list() # Test connection/model availability
        print(f"Backend: LLM Provider: Ollama ({OLLAMA_MODEL_NAME}) at {OLLAMA_BASE_URL}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ollama server or find model. Ensure Ollama is running and model '{OLLAMA_MODEL_NAME}' is downloaded. Error: {e}")
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Must be 'google' or 'ollama'.")


# --- Docling Converter Initialization ---
docling_converter = DocumentConverter()


# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://ai-interviewer-beta-six.vercel.app", # REPLACE with your actual Vercel URL
    "https://ai-interviewer-1-lj0z.onrender.com",
    "*" # TEMPORARY: Remove for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models for API Request/Response Bodies ---
# Note: Models are updated for a stateless flow. The client must now send the context.

class AnswerEvaluation(BaseModel):
    score: int = Field(..., ge=0, le=10)
    feedback: str = Field(..., min_length=10)

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str
    score: int
    feedback: str
    turn_num: int

class UploadResumeResponse(BaseModel):
    original_filename: str
    extracted_resume_text: str
    parsed_resume_data: dict

class StartInterviewRequest(BaseModel):
    extracted_resume_text: str
    job_role: str
    difficulty: str = "medium"

class StartInterviewResponse(BaseModel):
    initial_ai_message: str
    first_question: str
    total_questions: int = Field(12, description="Total number of main interview questions.")
    turn_num: int = Field(1, description="Current question number/turn.")

class SubmitAnswerRequest(BaseModel):
    extracted_resume_text: str # Context
    job_role: str # Context
    difficulty: str # Context
    qa_history: list[QuestionAnswerPair] # Context of the conversation so far
    current_question: str
    candidate_answer: str
    turn_num: int

class SubmitAnswerResponse(BaseModel):
    answered_question_evaluation: AnswerEvaluation
    next_question: str | None
    turn_num: int
    interview_complete: bool = False
    ai_closing_message: str | None = None

class GenerateSummaryRequest(BaseModel):
    job_role: str
    difficulty: str
    parsed_resume_data: dict
    qa_evaluations: list[QuestionAnswerPair]

class OverallSummaryResponse(BaseModel):
    summary: str = Field(..., min_length=50)


# --- Helper Function for LLM output cleaning/repair ---
def _repair_llm_json(response_content: str) -> dict:
    try:
        repaired_json = json_repair_loads(response_content)
        if not isinstance(repaired_json, dict) and not isinstance(repaired_json, list):
            raise ValueError("Repaired JSON is not a dictionary or list as expected.")
        return repaired_json
    except Exception as e:
        error_detail = f"JSON repair failed. Error: {e}. Raw response (partial): {response_content[:500]}..."
        raise ValueError(error_detail)


# --- Generic LLM Call Wrapper ---
async def _call_llm(prompt: str, is_json_output: bool = True, temperature: float = 0.0) -> str:
    global gemini_model, ollama_client, LLM_PROVIDER, GEMINI_MODEL_NAME, OLLAMA_MODEL_NAME

    if LLM_PROVIDER == "google":
        if not gemini_model:
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = await gemini_model.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        return response.text
    
    elif LLM_PROVIDER == "ollama":
        if not ollama_client:
            OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_client = OllamaClient(host=OLLAMA_BASE_URL)
        
        messages = [{"role": "user", "content": prompt}]
        options = {"temperature": temperature}

        response = ollama_client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=messages,
            options=options
        )
        return response['message']['content']
    
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be 'google' or 'ollama'.")


# --- GLOBAL INTERVIEW CONSTANTS ---
TECHNICAL_QUESTIONS_COUNT = 7
HR_QUESTIONS_COUNT = 5
MAX_QUESTIONS = TECHNICAL_QUESTIONS_COUNT + HR_QUESTIONS_COUNT # Total 12 main questions


# --- Internal LLM Helper for Question Generation / Fixed Pacing ---
async def _generate_next_question_llm(
    resume_text: str,
    job_role: str,
    difficulty: str,
    qa_history: list[dict], # Full history of {"question": "...", "answer": "...", "score": "...", "feedback": "..."}
    current_turn_num: int # This is 1-indexed for the questions (1 to 12)
) -> tuple[str | None, str | None]: # Returns (question_str | None, concluding_remark | None)
    
    if current_turn_num > MAX_QUESTIONS:
        closing_remark_prompt = f"""
        You are an AI interviewer. The interview is now complete after {MAX_QUESTIONS} questions.
        Generate a professional and courteous concluding remark. Do NOT ask another question.
        The remark should convey appreciation for their time and indicate that next steps will follow.
        Example: "Thank you for your time. We appreciate you interviewing with us. We will be in touch soon regarding next steps."
        Do NOT include any specific candidate name placeholder.
        """
        ai_concluding_message = await _call_llm(closing_remark_prompt, is_json_output=False, temperature=0.5)
        return None, ai_concluding_message

    question_type = ""
    question_guidance = ""
    
    if current_turn_num == 1:
        question_type = "initial introductory"
        question_guidance = "Ask a warm, standard introductory question like 'Please tell me about yourself.' based on their resume summary. This is the first question of the interview."
    elif current_turn_num <= (1 + TECHNICAL_QUESTIONS_COUNT):
        question_type = "technical"
        question_guidance = "Focus on core concepts, technologies, and problem-solving relevant to the job role and candidate's experience."
    else:
        question_type = "HR/behavioral"
        question_guidance = "Generate a scenario-based question that reflects a real-world working environment situation relevant to the job role. The question should probe soft skills, problem-solving in a team, handling conflicts, or prioritizing tasks. Do NOT ask generic 'tell me about yourself' or 'strengths/weaknesses' questions."

    history_str = ""
    if qa_history:
        history_str = "\n".join([
            f"Turn {i+1}:\nQuestion asked: {qa['question']}\nCandidate Answer: {qa['answer']}\nAI Feedback: {qa.get('feedback', 'N/A')}\n---"
            for i, qa in enumerate(qa_history)
        ])
        history_str = "\n\nPrevious Conversation History:\n" + history_str

    question_prompt = f"""
    You are an experienced human interviewer. Your goal is to generate the *next* interview question ({current_turn_num} of {MAX_QUESTIONS}) for a candidate.
    You must use their resume, the specified job role, difficulty level, and the previous conversation history.

    Rules for Question Generation:
    - Generate ONLY ONE question.
    - This question must be a {question_type} question. {question_guidance}
    - Tailor it directly to the candidate's experience, education, and skills from their resume.
    - Ensure it's relevant to the '{job_role}' role and '{difficulty}' difficulty.
    - **Adaptive Logic:**
        - If previous answers were vague, incorrect, or weak, ask a follow-up, a clarifying question, or simplify the next question related to that specific topic.
        - If previous answers were strong and comprehensive, ask a deeper, more challenging question on a new related topic, or move to the next logical area.
    - Do NOT repeat questions already asked.
    - The output MUST be a JSON object with a single key "question" which contains the question string. Do NOT include any other text or markdown fences.

    Example JSON format for regular question:
    {{"question": "Can you elaborate on your experience with distributed systems as mentioned in your resume, focusing on challenges you faced?"}}

    Interview Context:
        - Candidate's Resume Text (for reference):
        ---
        {resume_text}
        ---
        - Job Role: {job_role}
        - Difficulty: {difficulty}
        {history_str}

    Generate Question {current_turn_num} (of {MAX_QUESTIONS}):
    """
    
    llm_raw_response_content = await _call_llm(question_prompt, is_json_output=True, temperature=0.0)
    
    parsed_output = _repair_llm_json(llm_raw_response_content)
    generated_question = parsed_output.get("question")

    if not isinstance(generated_question, str) or not generated_question.strip():
        print(f"Warning: LLM failed to generate proper {question_type} question for turn {current_turn_num}. Raw response: {llm_raw_response_content}")
        # Fallback questions
        if current_turn_num == 1:
            return "Please tell me about yourself.", None
        elif current_turn_num <= (1 + TECHNICAL_QUESTIONS_COUNT):
             return "Could you describe your approach to optimizing code performance, or tell me about a technical project you worked on?", None
        else:
             return "Tell me about a time you had to overcome a significant challenge at work, how did you handle it?", None

    return generated_question.strip(), None

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/message")
async def send_message():
    return {"data":"Backend is Running"}


@app.post("/upload-resume/", response_model=UploadResumeResponse)
async def upload_resume(file: UploadFile = File(...)):
    try:
        fd, temp_file_path = tempfile.mkstemp(suffix=f".{file.filename.split('.')[-1]}")
        
        try:
            with os.fdopen(fd, 'wb') as tmp:
                content = await file.read()
                tmp.write(content)
            
            # Reset file pointer for fallback use if needed
            await file.seek(0)
            
            extracted_text_from_docling = ""
            doc = docling_converter.convert(temp_file_path)
            extracted_text_from_docling = doc.document.export_to_markdown()
                
        except Exception as e:
            file_content_fallback = await file.read() 
            if file.content_type == "application/pdf":
                try:
                    pdf_reader = PdfReader(io.BytesIO(file_content_fallback))
                    for page in pdf_reader.pages:
                        extracted_text_from_docling += page.extract_text() + "\n"
                except Exception as pdf_e:
                    raise HTTPException(status_code=400, detail=f"Error processing PDF with PyPDF: {pdf_e}")
            elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    document = Document(io.BytesIO(file_content_fallback))
                    for paragraph in document.paragraphs:
                        extracted_text_from_docling += paragraph.text + "\n"
                except Exception as docx_e:
                    raise HTTPException(status_code=400, detail=f"Error processing DOCX with python-docx: {docx_e}")
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type for fallback. Please upload a PDF or DOCX.")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path) 

        if not extracted_text_from_docling.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume. The file might be empty or unreadable.")

        extracted_resume_text = extracted_text_from_docling.strip()

        try: 
            parsing_prompt = f"""
            You are a highly skilled resume parser. Your task is to extract key information from the provided resume text and return it ONLY in a structured JSON format.
            The entire output must be a single, valid JSON object. Do NOT include any text, explanations, or markdown code fences (e.g., ```json) outside the JSON object itself.
            If a field is not found, omit it from the JSON.
            
            Extract the following fields:
            - name: (string) Full name of the candidate.
            - contact: (object) with email, phone, linkedin, github.
            - summary: (string) A brief professional summary or objective.
            - education: (array of objects) with degree, major, institution, start_date, end_date.
            - experience: (array of objects) with title, company, start_date, end_date, description.
            - skills: (object) with technical, soft, languages as arrays of strings.
            - projects: (array of objects) with name, description, link.

            Resume Text:
            ---
            {extracted_resume_text}
            ---
            """
            llm_raw_response_content = await _call_llm(parsing_prompt, is_json_output=True, temperature=0.0)
            parsed_resume_data = _repair_llm_json(llm_raw_response_content)

        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"AI parsing error (JSON repair): {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI parsing error (LLM API): {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

    # Return the processed data to the client instead of storing it
    return UploadResumeResponse(
        original_filename=file.filename,
        extracted_resume_text=extracted_resume_text,
        parsed_resume_data=parsed_resume_data
    )


@app.post("/start-interview/", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    # This endpoint no longer needs to access a database. It works with the provided data.
    intro_prompt = f"""
    You are an AI interviewer. Generate a warm, brief, and professional introductory message to start an interview.
    Mention that you will ask about their resume and the specific job role: '{request.job_role}'.
    End by asking them to "Please tell me about yourself."
    Do NOT include any other text or markdown.
    """
    ai_intro_message = await _call_llm(intro_prompt, is_json_output=False, temperature=0.7)

    return StartInterviewResponse(
        initial_ai_message=ai_intro_message,
        first_question="Please tell me about yourself.",
        total_questions=MAX_QUESTIONS,
        turn_num=1
    )


@app.post("/submit-answer/", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    # This endpoint is now stateless. It receives all context from the client.
    resume_text_context = request.extracted_resume_text
    current_job_role = request.job_role
    current_difficulty = request.difficulty
    qa_history_from_client = [item.model_dump() for item in request.qa_history]

    # Perform immediate evaluation for this answer
    evaluation_prompt = f"""
    You are an expert technical interviewer and strict performance evaluator. Your task is to objectively evaluate a candidate's answer to a single interview question.

    For the given question and candidate answer, you must:
    1.  **Assign a score** between 0 and 10. Be critical and objective.
        * **0-1: Completely Irrelevant/Empty.**
        * **2-3: Poor.** Major gaps or fundamentally incorrect.
        * **4-5: Fair/Below Average.** Shows some limited understanding, but lacks depth or clarity.
        * **6-7: Good/Average.** Solid understanding, mostly accurate.
        * **8-9: Very Good.** Comprehensive, accurate, and well-structured.
        * **10: Excellent.** Flawless, highly insightful, and demonstrates mastery.
    2.  **Provide concise, actionable feedback.** Explain the score and suggest improvements.
    3.  Consider the context of the candidate's resume, the job role ('{current_job_role}'), and difficulty ('{current_difficulty}').
    4.  The output MUST be a JSON object containing a "score" (integer) and "feedback" (string). Do NOT include any other text or markdown.

    Example JSON format for output:
    {{"score": 7, "feedback": "Good high-level overview, but lacked specific implementation details."}}

    ---
    Candidate's Resume Context:
    {resume_text_context}
    ---
    Current Question: {request.current_question}
    Candidate Answer: {request.candidate_answer}
    ---
    """
    llm_raw_response_content = await _call_llm(evaluation_prompt, is_json_output=True, temperature=0.0)
    
    try:
        parsed_evaluation = _repair_llm_json(llm_raw_response_content)
        current_evaluation = AnswerEvaluation(
            score=parsed_evaluation.get('score', 0),
            feedback=parsed_evaluation.get('feedback', 'Error parsing feedback.')
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"AI evaluation parsing error: {e}")

    # Append the newly evaluated answer to the history for the *next* question's context
    updated_qa_history = qa_history_from_client + [{
        "question": request.current_question,
        "answer": request.candidate_answer,
        "score": current_evaluation.score,
        "feedback": current_evaluation.feedback,
        "turn_num": request.turn_num
    }]

    # Generate the next question or conclude
    next_turn_num = request.turn_num + 1
    next_question_text, ai_concluding_message = await _generate_next_question_llm(
        resume_text=resume_text_context,
        job_role=current_job_role,
        difficulty=current_difficulty,
        qa_history=updated_qa_history,
        current_turn_num=next_turn_num
    )

    interview_is_complete = (next_question_text is None)

    return SubmitAnswerResponse(
        answered_question_evaluation=current_evaluation,
        next_question=next_question_text,
        turn_num=next_turn_num,
        interview_complete=interview_is_complete,
        ai_closing_message=ai_concluding_message
    )


@app.post("/generate-summary/", response_model=OverallSummaryResponse)
async def generate_summary(request: GenerateSummaryRequest):
    # This endpoint is now a POST to accept the full conversation history.
    try:
        candidate_name = request.parsed_resume_data.get('name', 'N/A Candidate')
        job_role = request.job_role
        difficulty = request.difficulty
        qa_evaluations = [item.model_dump() for item in request.qa_evaluations]
        resume_summary = request.parsed_resume_data.get('summary', 'No summary provided in resume.')

        if not qa_evaluations:
             raise HTTPException(status_code=400, detail="Cannot generate a summary with no question and answer data.")

        qa_feedback_str = "\n".join([
            f"Question: {item['question']}\nCandidate Answer: {item['answer']}\nAI Score: {item['score']}/10\nAI Feedback: {item['feedback']}\n---"
            for item in qa_evaluations
        ])

        overall_score = sum(item['score'] for item in qa_evaluations if isinstance(item.get('score'), (int, float))) / len(qa_evaluations)

        summary_prompt = f"""
        You are an expert HR professional. Generate a concise, objective overall summary for an interview session.
        The summary should:
        1.  Start with a brief overall assessment.
        2.  Highlight the candidate's key strengths.
        3.  Identify primary areas for improvement.
        4.  Provide actionable advice for the candidate.
        5.  Be a narrative, well-structured paragraph or two. Do NOT output JSON.
        6.  Mention the average score.

        Interview Context:
        - Candidate Name: {candidate_name}
        - Job Role: {job_role}
        - Difficulty: {difficulty}
        - Resume Summary: {resume_summary}
        - Average Score: {overall_score:.1f}/10

        Detailed Q&A History:
        ---
        {qa_feedback_str}
        ---
        Generate the overall summary and recommendations:
        """
        generated_summary = await _call_llm(summary_prompt, is_json_output=False, temperature=0.5)
        
        return OverallSummaryResponse(summary=generated_summary.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")