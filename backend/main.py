from datetime import datetime

from databases import Database
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from json_repair import loads as json_repair_loads
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB

import os
import io
import tempfile

import google.generativeai as genai
from ollama import Client as OllamaClient

from docling.document_converter import DocumentConverter


load_dotenv()

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

database = Database(DATABASE_URL)
metadata = MetaData()

# Define Database Tables
interview_sessions = Table(
    "interview_sessions",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("timestamp", DateTime, default=datetime.utcnow),
    Column("original_filename", String),
    Column("extracted_resume_text", Text), # Raw/Markdown text from Docling/fallback
    Column("parsed_resume_data", JSONB),  # Structured JSON from LLM parsing
    Column("job_role", String),
    Column("difficulty", String),
    Column("question_history", JSONB), # Stores list of {question, turn_num}
    Column("qa_evaluations", JSONB, default=[]), # Stores list of {question, answer, score, feedback, turn_num}
    Column("ai_intro_message", Text, default=""), # To store the AI's introductory message
    Column("ai_closing_message", Text, default="") # To store the AI's concluding message
)

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

# --- Database Connect/Disconnect Events ---
@app.on_event("startup")
async def startup():
    await database.connect()
    engine = create_engine(DATABASE_URL)
    metadata.create_all(engine)

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

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

class AnswerEvaluation(BaseModel):
    score: int = Field(..., ge=0, le=10)
    feedback: str = Field(..., min_length=10)

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str

class EvaluationRequest(BaseModel):
    session_id: int
    questions_with_answers: list[QuestionAnswerPair]

class EvaluationResponse(BaseModel):
    evaluations: list[AnswerEvaluation]

class StartInterviewRequest(BaseModel):
    session_id: int
    job_role: str
    difficulty: str = "medium"

class StartInterviewResponse(BaseModel):
    session_id: int
    initial_ai_message: str
    first_question: str
    total_questions: int = Field(12, description="Total number of main interview questions (excluding AI intro).")
    turn_num: int = Field(1, description="Current question number/turn.")


class SubmitAnswerRequest(BaseModel):
    session_id: int
    current_question: str
    candidate_answer: str
    turn_num: int

class SubmitAnswerResponse(BaseModel):
    session_id: int
    answered_question_evaluation: AnswerEvaluation
    next_question: str | None
    turn_num: int
    interview_complete: bool = False
    ai_closing_message: str | None = None

class OverallSummaryResponse(BaseModel):
    summary: str = Field(..., min_length=50)

# NEW: Pydantic Model for fetching full Q&A history
class FullQaHistoryResponse(BaseModel):
    qa_evaluations: list[dict] # List of dicts matching the stored qa_evaluations format
    ai_intro_message: str | None
    ai_closing_message: str | None
    job_role: str | None
    difficulty: str | None


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
    session_id: int,
    resume_text: str,
    job_role: str,
    difficulty: str,
    qa_history: list[dict], # Full history of {"question": "...", "answer": "...", "score": "...", "feedback": "..."}
    current_turn_num: int # This is 1-indexed for the questions (1 to 12)
) -> tuple[str | None, str | None]: # Returns (question_str | None, concluding_remark | None)
    
    # Interview concludes only after MAX_QUESTIONS are asked
    if current_turn_num > MAX_QUESTIONS:
        closing_remark_prompt = f"""
        You are an AI interviewer. The interview is now complete after {MAX_QUESTIONS} questions.
        Generate a professional and courteous concluding remark. Do NOT ask another question.
        The remark should convey appreciation for their time and indicate that next steps will follow.
        Example: "Thank you for your time. We appreciate you interviewing with us. We will be in touch soon regarding next steps."
        Do NOT include any specific candidate name placeholder.
        """
        ai_concluding_message = await _call_llm(closing_remark_prompt, is_json_output=False, temperature=0.5)
        return None, ai_concluding_message # Signal interview complete, return closing message

    # Determine question type (Technical or HR)
    question_type = ""
    question_guidance = ""
    
    # Logic for question types based on 1-indexed turnNum
    if current_turn_num == 1: # Turn 1 is always the "Tell me about yourself" question
        question_type = "initial introductory"
        question_guidance = "Ask a warm, standard introductory question like 'Please tell me about yourself.' based on their resume summary. This is the first question of the interview."
    elif current_turn_num <= (1 + TECHNICAL_QUESTIONS_COUNT): # Turn 2 to (1+7)=8 are technical
        question_type = "technical"
        question_guidance = "Focus on core concepts, technologies, and problem-solving relevant to the job role and candidate's experience."
    else: # Turn 9 to 12 are HR (after 1 intro + 7 technical)
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
        - Ensure variety in question types (technical, behavioral, situational, experience-based) over the course of the interview.
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
    concluding_remark = parsed_output.get("concluding_remark")

    if generated_question is None and isinstance(concluding_remark, str):
        return None, concluding_remark
    elif not isinstance(generated_question, str) or not generated_question.strip():
        print(f"Warning: LLM failed to generate proper {question_type} question for turn {current_turn_num} (Session ID: {session_id}). Raw response: {llm_raw_response_content}")
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
async def get_message():
    return {"data": "Data fetched successfully from FastAPI!"}

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        fd, temp_file_path = tempfile.mkstemp(suffix=f".{file.filename.split('.')[-1]}")
        
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(await file.read())
            
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
            raise HTTPException(status_code=400, detail="Could not extract text from the resume. The file might be empty or unreadable after Docling/fallback.")

        extracted_resume_text = extracted_text_from_docling.strip()

        llm_raw_response_content = ""
        try: 
            parsing_prompt = f"""
            You are a highly skilled resume parser. Your task is to extract key information from the provided resume text (which might be in Markdown format) and return it ONLY in a structured JSON format.

            Important Rules for JSON Output:
            1.  The entire output must be a single, valid JSON object.
            2.  Do NOT include any text, explanations, or markdown code fences (e.g., ```json) outside the JSON object itself.
            3.  All string values within the JSON must be properly escaped (e.g., double quotes within a string must be \\" or newlines as \\n).
            4.  Ensure all arrays and objects are correctly opened and closed.
            5.  All key-value pairs must be separated by commas, except for the last pair in an object or array.
            6.  If a field is not found, omit it from the JSON.

            Extract the following fields accurately:
            - name: (string) Full name of the candidate.
            - contact: (object)
                - email: (string) Candidate's email address.
                - phone: (string) Candidate's phone number.
                - linkedin: (string, optional) LinkedIn profile URL.
                - github: (string, optional) GitHub profile URL.
            - summary: (string, optional) A brief professional summary or objective.
            - education: (array of objects)
                - degree: (string) Degree obtained (e.g., "Master of Science", "B.Tech").
                - major: (string) Major field of study (e.g., "Computer Science").
                - institution: (string) Name of the university or college.
                - start_date: (string, optional) Start date (e.g., "YYYY-MM" or "Month YYYY").
                - end_date: (string, optional) End date (e.g., "YYYY-MM" or "Month YYYY", "Present" if currently enrolled).
            - experience: (array of objects)
                - title: (string) Job title.
                - company: (string) Company name.
                - start_date: (string) Start date of employment (e.g., "YYYY-MM" or "Month YYYY").
                - end_date: (string, optional) End date of employment (e.g., "YYYY-MM" or "Month YYYY", "Present" if current).
                - description: (string) A detailed description of responsibilities and achievements. Preserve newlines and special characters by escaping them correctly within the JSON string.
            - skills: (object)
                - technical: (array of strings) Programming languages, frameworks, tools, databases, cloud platforms.
                - soft: (array of strings, optional) Communication, teamwork, problem-solving, leadership.
                - languages: (array of strings, optional) Spoken languages.
            - projects: (array of objects, optional)
                - name: (string) Project name.
                - description: (string) Project description and technologies used. Preserve newlines and special characters by escaping them correctly within the JSON string.
                - link: (string, optional) Link to the project (e.g., GitHub repo).

            Resume Text:
            ---
            {extracted_resume_text}
            ---
            """
            llm_raw_response_content = await _call_llm(parsing_prompt, is_json_output=True, temperature=0.0)

            parsed_resume_data = _repair_llm_json(llm_raw_response_content)

            query = interview_sessions.insert().values(
                original_filename=file.filename,
                extracted_resume_text=extracted_resume_text,
                parsed_resume_data=parsed_resume_data,
                job_role="",
                difficulty="",
                question_history=[],
                qa_evaluations=[]
            )
            last_record_id = await database.execute(query)

        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"AI parsing error (JSON repair): {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI parsing error (LLM API) or DB insert error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

    return {"message": "Resume uploaded and processed. Ready for question generation.", "session_id": last_record_id}


# Old /generate-questions/ endpoint is effectively replaced by /start-interview/ and _generate_next_question_llm.
# Keeping it commented out to avoid confusion and ensure correct flow.
# @app.post("/generate-questions/")
# async def generate_questions_deprecated(...):
#     raise HTTPException(status_code=405, detail="Use /start-interview/ and /submit-answer/ for conversational flow.")


@app.post("/start-interview/", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    query = interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    raw_session_data = await database.fetch_one(query)
    session_data = dict(raw_session_data)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    # Generate AI Introduction message
    intro_prompt = f"""
    You are an AI interviewer. Generate a warm, brief, and professional introductory message to start an interview.
    Mention that you will ask about their resume and the specific job role: '{request.job_role}'.
    End by asking them to "Please tell me about yourself."
    Do NOT include any other text or markdown.
    """
    ai_intro_message = await _call_llm(intro_prompt, is_json_output=False, temperature=0.7)

    # Update AI intro message in DB
    update_query = interview_sessions.update().where(interview_sessions.c.id == request.session_id).values(
        job_role=request.job_role,
        difficulty=request.difficulty,
        question_history=[], # Reset history for a new interview
        qa_evaluations=[],   # Reset evaluations for a new interview
        ai_intro_message=ai_intro_message # Store the AI's intro message
    )
    await database.execute(update_query)

    # The first question (Turn 1) is "Please tell me about yourself."
    return StartInterviewResponse(
        session_id=request.session_id,
        initial_ai_message=ai_intro_message,
        first_question="Please tell me about yourself.", # This will be the first user question displayed (Turn 1)
        total_questions=MAX_QUESTIONS, # Total main questions (from global constant)
        turn_num=1 # Start question numbering at 1 for "Tell me about yourself"
    )


@app.post("/submit-answer/", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    query = interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    raw_session_data = await database.fetch_one(query)
    session_data = dict(raw_session_data)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    resume_text_context = session_data["extracted_resume_text"]
    current_job_role = session_data["job_role"]
    current_difficulty = session_data["difficulty"]
    qa_history_from_db = session_data.get("qa_evaluations", []) # Use existing QA history for context
    
    # Perform immediate evaluation for this answer
    evaluation_prompt = f"""
    You are an expert technical interviewer and strict performance evaluator. Your task is to objectively and rigorously evaluate a candidate's answer to a single interview question.

    For the given question and candidate answer, you must:
    1.  **Assign a score** between 0 and 10. Be critical and objective.
        * **0-1: Completely Irrelevant/Empty/Nonsense.** Answer provides no value or is just a single character/word like "a".
        * **2-3: Poor.** Major gaps, fundamentally incorrect, completely off-topic, or very minimal understanding demonstrated.
        * **4-5: Fair/Below Average.** Shows some limited understanding, but lacks depth, clarity, accuracy, or is too generic. May contain inaccuracies.
        * **6-7: Good/Average.** Solid understanding, mostly accurate, clear, and relevant. Could be more detailed or comprehensive.
        * **8-9: Very Good.** Comprehensive, accurate, well-structured, insightful, strong relevant examples. Demonstrates strong grasp.
        * **10: Excellent.** Flawless, highly insightful, demonstrates mastery, exceptional clarity and conciseness, goes beyond expectations.

    2.  **Provide concise, actionable feedback** for each answer.
        * Be direct and honest about weaknesses.
        * Explain *why* a particular score was given.
        * Suggest specific areas for improvement.
        * Do NOT be overly generous. If an answer is bad, state it clearly.

    3.  Consider the context of the **candidate's original resume** (provided below, potentially in Markdown) and the **job role** ('{current_job_role}') and **difficulty** ('{current_difficulty}') of the interview. Use the resume to check for factual accuracy or relevance to their claimed experience.

    4.  The output MUST be a JSON object containing a "score" (integer) and "feedback" (string). Do NOT include any other text or markdown fences.

    Example JSON format for output:
    {{"score": 7, "feedback": "Good high-level overview, but lacked specific implementation details or examples from your experience."}}

    ---
    Candidate's Resume Context (for reference):
    {resume_text_context}
    ---

    Current Question: {request.current_question}
    Candidate Answer: {request.candidate_answer}
    ---
    """
    llm_raw_response_content = await _call_llm(evaluation_prompt, is_json_output=True, temperature=0.0)
    
    try: # Inner try for parsing evaluation
        parsed_evaluation = _repair_llm_json(llm_raw_response_content)
        
        if not isinstance(parsed_evaluation, dict) or "score" not in parsed_evaluation or "feedback" not in parsed_evaluation:
            raise ValueError("LLM evaluation response not in expected JSON format (missing score/feedback).")
        
        current_evaluation = AnswerEvaluation(
            score=parsed_evaluation['score'],
            feedback=parsed_evaluation['feedback']
        )

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"AI evaluation parsing error (JSON): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI evaluation API error: {e}")

    # 1. Store this turn's Q&A and evaluation
    updated_qa_evaluations = qa_history_from_db + [{
        "question": request.current_question,
        "answer": request.candidate_answer,
        "score": current_evaluation.score,
        "feedback": current_evaluation.feedback,
        "turn_num": request.turn_num
    }]
    
    # 2. Generate the next question (or concluding remark)
    next_turn_num = request.turn_num + 1
    next_question_text, ai_concluding_message = None, None # Initialize
    
    # Check if we should generate a next question or conclude the interview
    if request.turn_num < MAX_QUESTIONS: # If current turn is less than MAX_QUESTIONS, generate next question
        next_question_result = await _generate_next_question_llm(
            session_id=request.session_id,
            resume_text=resume_text_context,
            job_role=current_job_role,
            difficulty=current_difficulty,
            qa_history=updated_qa_evaluations, # Pass updated QA history for adaptive questioning
            current_turn_num=next_turn_num # Pass the next turn number (2 for first real Q, etc.)
        )
        next_question_text, ai_concluding_message = next_question_result
    else: # If current turn is MAX_QUESTIONS, the interview is complete. Get closing message.
        interview_is_complete = True
        # Call _generate_next_question_llm one last time specifically to get the closing message
        # by forcing current_turn_num > MAX_QUESTIONS condition.
        _, ai_concluding_message = await _generate_next_question_llm(
            session_id=request.session_id,
            resume_text=resume_text_context,
            job_role=current_job_role,
            difficulty=current_difficulty,
            qa_history=updated_qa_evaluations,
            current_turn_num=MAX_QUESTIONS + 1 # This will trigger the "interview complete" branch in _generate_next_question_llm
        )

    interview_is_complete = (next_question_text is None) # Confirm based on generated_question

    # Update question_history in DB
    current_session_data_for_history_update = dict(await database.fetch_one(
        interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    ))
    updated_question_history_db = current_session_data_for_history_update.get("question_history", [])

    if not interview_is_complete: # Only append if interview is not complete
        updated_question_history_db.append({"question": next_question_text, "turn_num": next_turn_num})
    # If interview is complete, a closing message is already generated or handled.

    update_query = interview_sessions.update().where(interview_sessions.c.id == request.session_id).values(
        qa_evaluations=updated_qa_evaluations,
        question_history=updated_question_history_db,
        ai_closing_message=ai_concluding_message if ai_concluding_message else "" # Store closing message
    )
    await database.execute(update_query)

    return SubmitAnswerResponse(
        session_id=request.session_id,
        answered_question_evaluation=current_evaluation,
        next_question=next_question_text,
        turn_num=next_turn_num,
        interview_complete=interview_is_complete,
        ai_closing_message=ai_concluding_message # Return closing message
    )


@app.get("/generate-summary/{session_id}", response_model=OverallSummaryResponse)
async def generate_summary(session_id: int):
    try:
        query = interview_sessions.select().where(interview_sessions.c.id == session_id)
        raw_session_data = await database.fetch_one(query)
        session_data = dict(raw_session_data)

        if not session_data:
            raise HTTPException(status_code=404, detail="Interview session not found for summary generation.")

        candidate_name = session_data['parsed_resume_data'].get('name', 'N/A Candidate')
        job_role = session_data.get('job_role', 'N/A Job Role')
        difficulty = session_data.get('difficulty', 'N/A Difficulty')
        
        qa_evaluations = session_data.get('qa_evaluations', []) 
        
        resume_summary = session_data['parsed_resume_data'].get('summary', 'No summary provided in resume.')

        qa_feedback_str = "\n".join([
            f"Question: {item['question']}\nCandidate Answer: {item['answer']}\nAI Score: {item['score']}/10\nAI Feedback: {item['feedback']}\n---"
            for item in qa_evaluations
        ])

        overall_score = sum(item['score'] for item in qa_evaluations if isinstance(item.get('score'), (int, float))) / len(qa_evaluations) if qa_evaluations else 0

        summary_prompt = f"""
        You are an expert HR professional and interview analyst. Your task is to generate a concise, objective overall summary and feedback for an interview session based on the provided data.

        The summary should:
        1.  Start with a brief overall assessment (e.g., "The candidate demonstrated X, but needs to improve on Y.").
        2.  Highlight the candidate's key strengths based on their answers and scores.
        3.  Identify primary areas for improvement, referring to specific questions or themes.
        4.  Provide actionable advice for the candidate to improve their interview performance.
        5.  Be a narrative, well-structured paragraph or two. Do NOT output JSON for this. Just the plain text summary.
        6.  Mention the average score.

        Interview Context:
        - Candidate Name: {candidate_name}
        - Job Role: {job_role}
        - Difficulty: {difficulty}
        - Resume Summary: {resume_summary}
        - Average Score: {overall_score:.1f}/10

        Detailed Questions, Answers, and AI Feedback:
        ---
        {qa_feedback_str}
        ---

        Generate the overall summary and recommendations:
        """
        generated_summary = await _call_llm(summary_prompt, is_json_output=False, temperature=0.5)
        
        return OverallSummaryResponse(summary=generated_summary.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate overall summary due to data retrieval or LLM error: {e}")