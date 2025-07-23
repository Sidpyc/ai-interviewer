from datetime import datetime

from databases import Database
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# FPDF is not used in the current version.
# from fpdf import FPDF
from json_repair import loads as json_repair_loads
# Commenting out Ollama client import for Google Gemini integration
# from ollama import Client as OllamaClient
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB

import os
import io
import tempfile

from docling.document_converter import DocumentConverter
# NEW: Google Generative AI SDK import
import google.generativeai as genai


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
    Column("extracted_resume_text", Text),
    Column("parsed_resume_data", JSONB),
    Column("job_role", String),
    Column("difficulty", String),
    Column("generated_questions", JSONB),
    Column("qa_evaluations", JSONB, default={}),
)

# --- LLM Client Configuration (Now for Google Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Choose your Gemini model:
# gemini-pro is a good general-purpose model.
# gemini-1.5-flash is faster and good for large contexts.
# gemini-1.5-pro is more capable but slower and potentially more expensive.
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Recommended for speed and cost-effectiveness


# Commenting out Ollama client initialization
# ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3")
# ollama_client = OllamaClient(host=ollama_base_url)


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
    "https://ai-interviewer-opqjz6ke6-sidpycs-projects.vercel.app",
    "https://ai-interviewer-beta-six.vercel.app/"

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Request/Response Bodies ---

class QuestionGenerationRequest(BaseModel):
    session_id: int
    job_role: str
    difficulty: str = "medium"

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str

class EvaluationRequest(BaseModel):
    session_id: int
    questions_with_answers: list[QuestionAnswerPair]

class AnswerEvaluation(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Score for the answer on a scale of 0 to 10.")
    feedback: str = Field(..., min_length=10, description="Constructive feedback for the answer.")

class EvaluationResponse(BaseModel):
    evaluations: list[AnswerEvaluation]

class OverallSummaryResponse(BaseModel):
    summary: str = Field(..., min_length=50, description="Overall interview summary and feedback.")


# --- Helper Function for LLM output cleaning/repair ---
# This function is still crucial for handling potential non-JSON output from Gemini
def _repair_llm_json(response_content: str) -> dict:
    try:
        repaired_json = json_repair_loads(response_content)
        if not isinstance(repaired_json, dict) and not isinstance(repaired_json, list):
            raise ValueError("Repaired JSON is not a dictionary or list as expected.")
        return repaired_json
    except Exception as e:
        error_detail = f"JSON repair failed. Error: {e}. Raw response (partial): {response_content[:500]}..."
        raise ValueError(error_detail)


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/message")
async def get_message():
    return {"data": "Data fetched successfully from FastAPI!"}

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try: # Outer try for entire endpoint logic (Docling, LLM parsing, DB insert)
        fd, temp_file_path = tempfile.mkstemp(suffix=f".{file.filename.split('.')[-1]}")
        
        try: # Inner try for Docling conversion and fallback file handling
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(await file.read())
            
            extracted_text_from_docling = ""
            
            doc = docling_converter.convert(temp_file_path)
            extracted_text_from_docling = doc.document.export_to_markdown()
                
        except Exception as e: # Catch exceptions from Docling conversion
            # Fallback to simple pypdf/python-docx extraction if Docling fails
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
        finally: # This finally block ensures temp file is removed after inner try completes
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path) 

        if not extracted_text_from_docling.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume. The file might be empty or unreadable after Docling/fallback.")

        extracted_resume_text = extracted_text_from_docling.strip()

        llm_raw_response_content = ""
        # Inner try block for LLM parsing and DB insert
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
            # Using Google Gemini (instead of Ollama) for parsing
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = await model.generate_content_async(
                parsing_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            llm_raw_response_content = response.text # Gemini's content is in .text

            parsed_resume_data = _repair_llm_json(llm_raw_response_content)

            query = interview_sessions.insert().values(
                original_filename=file.filename,
                extracted_resume_text=extracted_resume_text,
                parsed_resume_data=parsed_resume_data,
                job_role="",
                difficulty="",
                generated_questions=[],
                qa_evaluations={}
            )
            last_record_id = await database.execute(query)

        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"AI parsing error (Gemini JSON): {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI parsing error (Gemini API) or DB insert error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

    return {"message": "Resume uploaded and processed. Ready for question generation.", "session_id": last_record_id}

@app.post("/generate-questions/")
async def generate_questions(request: QuestionGenerationRequest):
    query = interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    raw_session_data = await database.fetch_one(query)
    session_data = dict(raw_session_data)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Interview session not found. Please upload a resume first.")

    resume_text_for_questions = session_data["extracted_resume_text"]
    
    job_role = request.job_role
    difficulty = request.difficulty

    question_prompt = f"""
    You are an experienced human interviewer. Your goal is to generate interview questions for a candidate based on their resume and a specified job role and difficulty level.

    Important Rules for JSON Output:
    1.  The entire output must be a single, valid JSON object with a key "questions" which contains a JSON array of strings.
    2.  Do NOT include any text, explanations, or markdown code fences (e.g., ```json) outside the JSON object itself.
    3.  All string values within the JSON must be properly escaped (e.g., double quotes within a string must be \\" or newlines as \\n).
    4.  Ensure all arrays and objects are correctly opened and closed.
    5.  All key-value pairs must be separated by commas, except for the last pair in an object or array.

    Example JSON format:
    {{"questions": ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?", "Question 6?", "Question 7?", "Question 8?", "Question 9?", "Question 10?"]}}

    The questions should be:
    - Tailored directly to the candidate's experience, education, and skills mentioned in their resume.
    - Relevant to the '{job_role}' role.
    - Adjusted for a '{difficulty}' difficulty level (e.g., for 'easy', ask foundational questions; for 'medium', ask moderate technical/behavioral; for 'hard', ask in-depth technical or complex problem-solving).
    - Aim for exactly 10 questions.
    - Ensure a mix of technical, behavioral, and experience-based questions.

    Candidate's Resume Text (may include markdown formatting from document conversion):
    ---
    {resume_text_for_questions}
    ---

    Job Role: {job_role}
    Difficulty: {difficulty}
    """
    llm_raw_response_content = ""
    try:
        # Using Google Gemini (instead of Ollama) for question generation
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = await model.generate_content_async(
            question_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        llm_raw_response_content = response.text 
        
        parsed_llm_output = _repair_llm_json(llm_raw_response_content)
        generated_questions = parsed_llm_output.get("questions", [])

        if not isinstance(generated_questions, list):
            raise ValueError("LLM response 'questions' field is not a list after parsing. Expected JSON with 'questions' array.")

        update_query = interview_sessions.update().where(interview_sessions.c.id == request.session_id).values(
            job_role=job_role,
            difficulty=difficulty,
            generated_questions=generated_questions
        )
        await database.execute(update_query)

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"AI question generation error (Gemini JSON): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI question generation error (Gemini API): {e}")

    return {
        "job_role": job_role,
        "difficulty": difficulty,
        "questions": generated_questions
    }


@app.post("/evaluate-answers/", response_model=EvaluationResponse)
async def evaluate_answers(request: EvaluationRequest):
    query = interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    raw_session_data = await database.fetch_one(query)
    session_data = dict(raw_session_data)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Interview session not found. Please upload a resume and generate questions first.")

    resume_text_context = session_data["extracted_resume_text"]
    
    qa_pairs_str = "\n".join([
        f"Question {i+1}: {qa.question}\nAnswer {i+1}: {qa.answer}\n---"
        for i, qa in enumerate(request.questions_with_answers)
    ])

    evaluation_prompt = f"""
    You are an expert technical interviewer and strict performance evaluator. Your task is to objectively and rigorously evaluate a candidate's answers to a set of interview questions.

    For each question and its corresponding answer, you must:
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

    3.  Consider the context of the **candidate's original resume** (provided below, potentially in Markdown) and the **job role** they are being interviewed for (implicitly understood from the questions generated). Use the resume to check for factual accuracy or relevance to their claimed experience.

    4.  The output MUST be a JSON array of objects, where each object contains a "score" (integer) and "feedback" (string) for the corresponding question. The order of evaluations must match the order of questions provided.

    Example JSON format:
    {{
        "evaluations": [
            {{"score": 2, "feedback": "Answer was completely irrelevant to the question asked. Lacks any technical detail required for this role."}},
            {{"score": 7, "feedback": "Good high-level overview, but lacked specific implementation details or examples from your experience."}},
            {{"score": 1, "feedback": "Answer was a single letter 'a'. This is not a valid response."}},
            // ... for all 10 questions
        ]
    }}

    ---
    Candidate's Resume Context (for reference, potentially Markdown):
    {resume_text_context}
    ---

    Candidate's Questions and Answers:
    {qa_pairs_str}
    ---
    """
    llm_raw_response_content = ""
    try:
        # NEW: Google Gemini API Call for Evaluation
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = await model.generate_content_async(
            evaluation_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        llm_raw_response_content = response.text

        evaluation_result_data = _repair_llm_json(llm_raw_response_content)
        
        if not isinstance(evaluation_result_data, dict) or "evaluations" not in evaluation_result_data or not isinstance(evaluation_result_data["evaluations"], list):
            raise ValueError("LLM response for evaluation is not in the expected JSON format (missing 'evaluations' array).")

        parsed_evaluations_list = [
            {"question": qa.question, "answer": qa.answer, "score": eval_item['score'], "feedback": eval_item['feedback']}
            for qa, eval_item in zip(request.questions_with_answers, evaluation_result_data["evaluations"])
        ]
        
        update_query = interview_sessions.update().where(interview_sessions.c.id == request.session_id).values(
            qa_evaluations=parsed_evaluations_list
        )
        await database.execute(update_query)

        return EvaluationResponse(evaluations=[
            AnswerEvaluation(score=item['score'], feedback=item['feedback']) 
            for item in parsed_evaluations_list
        ])

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"AI evaluation error (Gemini JSON): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI evaluation error (Gemini API): {e}")


# --- Endpoint for Overall Interview Summary ---
@app.get("/generate-summary/{session_id}", response_model=OverallSummaryResponse)
async def generate_summary(session_id: int):
    try: # Outer try for entire endpoint logic (DB fetch, LLM summary generation)
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
        llm_raw_response_content = ""
        try: # Inner try for Ollama API call
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = await model.generate_content_async(
                summary_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.5)
            )
            llm_raw_response_content = response.text
            
            generated_summary = llm_raw_response_content.strip()

            return OverallSummaryResponse(summary=generated_summary)

        except Exception as e: # This 'except' catches errors from the inner Ollama call
            raise HTTPException(status_code=500, detail=f"AI summary generation error: {e}")

    except Exception as e: # This 'except' catches errors from the outer try block (e.g., DB fetch)
        raise HTTPException(status_code=500, detail=f"Failed to generate overall summary due to data retrieval error: {e}")