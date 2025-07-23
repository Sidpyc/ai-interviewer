import io
import json
import os
import re
import tempfile 
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Response # Import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ollama import Client as OllamaClient
from json_repair import loads as json_repair_loads

# New import for PDF generation
from fpdf import FPDF # fpdf2 is imported as FPDF

# For Docling
from docling.document_converter import DocumentConverter

# For PDF and DOCX fallback
from pypdf import PdfReader
from docx import Document

# Database imports
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB


load_dotenv()

# --- Database Configuration (ensure this is correct from your .env) ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

database = Database(DATABASE_URL)
metadata = MetaData()

# Define Database Tables (ensure match with your DB schema)
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
    Column("qa_evaluations", JSONB, default={}), # List of {question, answer, score, feedback}
)

# --- LLM Client Configuration ---
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3") 

print(f"Backend: Initializing Ollama Client at {ollama_base_url} with model {ollama_model_name}")
ollama_client = OllamaClient(host=ollama_base_url)

# --- Docling Converter Initialization ---
print("Backend: Initializing Docling DocumentConverter...")
docling_converter = DocumentConverter()
print("Backend: Docling DocumentConverter initialized.")


# --- FastAPI App Initialization ---
app = FastAPI()

# --- Database Connect/Disconnect Events ---
@app.on_event("startup")
async def startup():
    print("Backend: Connecting to database...")
    await database.connect()
    engine = create_engine(DATABASE_URL)
    metadata.create_all(engine) # This creates tables if they don't exist
    print("Backend: Database connected and tables ensured.")

@app.on_event("shutdown")
async def shutdown():
    print("Backend: Disconnecting from database...")
    await database.disconnect()
    print("Backend: Database disconnected.")

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
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


# --- Helper Function for LLM output cleaning/repair ---
def _repair_llm_json(response_content: str) -> dict:
    print(f"Backend: Attempting to repair JSON from LLM response. Raw start: {response_content[:100]}...")
    try:
        repaired_json = json_repair_loads(response_content)
        if not isinstance(repaired_json, dict) and not isinstance(repaired_json, list):
            raise ValueError("Repaired JSON is not a dictionary or list as expected.")
        print("Backend: JSON repair successful.")
        return repaired_json
    except Exception as e:
        error_detail = f"JSON repair failed. Error: {e}. Raw response (partial): {response_content[:500]}..."
        print(f"Backend: JSON repair FAILED: {error_detail}")
        raise ValueError(error_detail)


# --- API Endpoints ---
@app.get("/")
async def read_root():
    print("Backend: Root endpoint accessed.")
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/message")
async def get_message():
    print("Backend: API message endpoint accessed.")
    return {"data": "Data fetched successfully from FastAPI!"}

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    print(f"Backend: /upload-resume/ endpoint called for file: {file.filename}")
    
    fd, temp_file_path = tempfile.mkstemp(suffix=f".{file.filename.split('.')[-1]}")
    try:
        print(f"Backend: Saving uploaded file to temporary path: {temp_file_path}")
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(await file.read())
        print("Backend: File saved. Starting Docling conversion...")
        
        extracted_text_from_docling = ""
        
        doc = docling_converter.convert(temp_file_path)
        extracted_text_from_docling = doc.document.export_to_markdown()
        print("Backend: Docling conversion complete. Extracted Markdown.")
            
    except Exception as e:
        print(f"Backend: Docling conversion FAILED: {e}. Attempting fallback text extraction...")
        file_content_fallback = await file.read() 
        if file.content_type == "application/pdf":
            try:
                pdf_reader = PdfReader(io.BytesIO(file_content_fallback))
                for page in pdf_reader.pages:
                    extracted_text_from_docling += page.extract_text() + "\n"
                print("Backend: Fallback PDF extraction successful.")
            except Exception as pdf_e:
                print(f"Backend: Fallback PDF extraction FAILED: {pdf_e}")
                raise HTTPException(status_code=400, detail=f"Error processing PDF with PyPDF: {pdf_e}")
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                document = Document(io.BytesIO(file_content_fallback))
                for paragraph in document.paragraphs:
                    extracted_text_from_docling += paragraph.text + "\n"
                print("Backend: Fallback DOCX extraction successful.")
            except Exception as docx_e:
                print(f"Backend: Fallback DOCX extraction FAILED: {docx_e}")
                raise HTTPException(status_code=400, detail=f"Error processing DOCX with python-docx: {docx_e}")
        else:
            print("Backend: Unsupported file type for fallback.")
            raise HTTPException(status_code=400, detail="Unsupported file type for fallback. Please upload a PDF or DOCX.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) 
            print(f"Backend: Temporary file {temp_file_path} removed.")

    if not extracted_text_from_docling.strip():
        print("Backend: No text extracted after Docling/fallback.")
        raise HTTPException(status_code=400, detail="Could not extract text from the resume. The file might be empty or unreadable after Docling/fallback.")

    extracted_resume_text = extracted_text_from_docling.strip()
    print(f"Backend: Extracted text length: {len(extracted_resume_text)} chars. Starting LLM parsing...")

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
        print(f"Backend: Sending parsing prompt to Ollama model {ollama_model_name}...")
        response = ollama_client.chat(
            model=ollama_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant designed to output JSON."},
                {"role": "user", "content": parsing_prompt}
            ],
            options={"temperature": 0.0}
        )
        llm_raw_response_content = response['message']['content']
        print(f"Backend: Ollama response received. Length: {len(llm_raw_response_content)} chars. Attempting JSON repair.")

        parsed_resume_data = _repair_llm_json(llm_raw_response_content)
        print("Backend: Resume parsing by LLM complete.")

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
        print(f"Backend: Resume data stored in DB with session_id: {last_record_id}")

    except ValueError as e:
        print(f"Backend: AI parsing failed (ValueError): {e}")
        raise HTTPException(status_code=500, detail=f"AI parsing error (Ollama): {e}")
    except Exception as e:
        print(f"Backend: General error during Ollama API call or DB operation for parsing: {e}")
        raise HTTPException(status_code=500, detail=f"AI parsing error (Ollama): {e}")

    print("Backend: /upload-resume/ processing finished. Returning session ID.")
    return {"message": "Resume uploaded and processed. Ready for question generation.", "session_id": last_record_id}

@app.post("/generate-questions/")
async def generate_questions(request: QuestionGenerationRequest):
    print(f"Backend: /generate-questions/ endpoint called for session_id: {request.session_id}, role '{request.job_role}', difficulty '{request.difficulty}'")

    query = interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    session_data = await database.fetch_one(query)
    
    if not session_data:
        print(f"Backend: Error: Session ID {request.session_id} not found.")
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
    print(f"Backend: Sending question generation prompt to Ollama model {ollama_model_name}...")
    llm_raw_response_content = ""
    try:
        response = ollama_client.chat(
            model=ollama_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant designed to output JSON."},
                {"role": "user", "content": question_prompt}
            ],
            options={"temperature": 0.0}
        )
        llm_raw_response_content = response['message']['content']
        print(f"Backend: Ollama response for questions received. Length: {len(llm_raw_response_content)} chars. Attempting JSON repair.")

        parsed_llm_output = _repair_llm_json(llm_raw_response_content)
        generated_questions = parsed_llm_output.get("questions", [])

        if not isinstance(generated_questions, list):
            print(f"Backend: Validation failed: 'questions' field in LLM response is not a list.")
            raise ValueError("LLM response 'questions' field is not a list after parsing. Expected JSON with 'questions' array.")

        print("Backend: Question generation by LLM complete.")

        update_query = interview_sessions.update().where(interview_sessions.c.id == request.session_id).values(
            job_role=job_role,
            difficulty=difficulty,
            generated_questions=generated_questions
        )
        await database.execute(update_query)
        print(f"Backend: Updated session {request.session_id} with questions.")

    except ValueError as e:
        print(f"Backend: AI question generation failed (ValueError): {e}")
        raise HTTPException(status_code=500, detail=f"AI question generation error (Ollama): {e}")
    except Exception as e:
        print(f"Backend: General error during Ollama API call or DB operation for questions: {e}")
        raise HTTPException(status_code=500, detail=f"AI question generation error (Ollama): {e}")

    print("Backend: /generate-questions/ processing finished. Returning questions.")
    return {
        "job_role": job_role,
        "difficulty": difficulty,
        "questions": generated_questions
    }


@app.post("/evaluate-answers/", response_model=EvaluationResponse)
async def evaluate_answers(request: EvaluationRequest):
    print(f"Backend: /evaluate-answers/ endpoint called for session_id: {request.session_id}.")

    query = interview_sessions.select().where(interview_sessions.c.id == request.session_id)
    session_data = await database.fetch_one(query)
    
    if not session_data:
        print(f"Backend: Error: Session ID {request.session_id} not found for evaluation.")
        raise HTTPException(status_code=404, detail="Interview session not found. Please upload a resume and generate questions first.")

    resume_text_context = session_data["extracted_resume_text"] # Context for LLM
    
    qa_pairs_str = "\n".join([
        f"Question {i+1}: {qa.question}\nAnswer {i+1}: {qa.answer}\n---"
        for i, qa in enumerate(request.questions_with_answers)
    ])
    print(f"Backend: Received {len(request.questions_with_answers)} Q&A pairs for evaluation.")

    evaluation_prompt = f"""
    You are an expert technical interviewer and performance evaluator. Your task is to evaluate a candidate's answers to a set of interview questions.

    For each question and its corresponding answer, you must:
    1.  **Assign a score** between 0 and 10, where:
        * 0-3: Poor (Significant gaps, incorrect, or irrelevant)
        * 4-6: Fair (Some understanding, but lacks depth, clarity, or accuracy)
        * 7-8: Good (Solid understanding, mostly accurate, clear)
        * 9-10: Excellent (Comprehensive, accurate, well-structured, insightful, strong example usage)
    2.  **Provide concise, constructive feedback** for each answer. Focus on what was good, what could be improved, and why the given score was assigned.
    3.  Consider the context of the **candidate's original resume** (provided below, potentially in Markdown) and the **job role** they are being interviewed for (implicitly understood from the questions generated).
    4.  The output MUST be a JSON array of objects, where each object contains a "score" (integer) and "feedback" (string) for the corresponding question. The order of evaluations must match the order of questions provided.

    Example JSON format for output:
    {{
        "evaluations": [
            {{"score": 8, "feedback": "Good explanation of X, but could add detail on Y."}},
            {{"score": 5, "feedback": "Answer was too generic. Lacked specific examples."}},
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
    print(f"Backend: Sending evaluation prompt to Ollama model {ollama_model_name}...")
    llm_raw_response_content = ""
    try:
        response = ollama_client.chat(
            model=ollama_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant designed to output JSON."},
                {"role": "user", "content": evaluation_prompt}
            ],
            options={"temperature": 0.0}
        )
        llm_raw_response_content = response['message']['content']
        print(f"Backend: Ollama response for evaluation received. Length: {len(llm_raw_response_content)} chars. Attempting JSON repair.")

        evaluation_result_data = _repair_llm_json(llm_raw_response_content)
        
        if not isinstance(evaluation_result_data, dict) or "evaluations" not in evaluation_result_data or not isinstance(evaluation_result_data["evaluations"], list):
            print(f"Backend: Validation failed: 'evaluations' field in LLM response is not a list or missing.")
            raise ValueError("LLM response for evaluation is not in the expected JSON format (missing 'evaluations' array).")

        parsed_evaluations_list = [
            {"question": qa.question, "answer": qa.answer, "score": eval_item['score'], "feedback": eval_item['feedback']}
            for qa, eval_item in zip(request.questions_with_answers, evaluation_result_data["evaluations"])
        ]
        
        update_query = interview_sessions.update().where(interview_sessions.c.id == request.session_id).values(
            qa_evaluations=parsed_evaluations_list
        )
        await database.execute(update_query)
        print(f"Backend: Updated session {request.session_id} with evaluations.")

        return EvaluationResponse(evaluations=[
            AnswerEvaluation(score=item['score'], feedback=item['feedback']) 
            for item in parsed_evaluations_list
        ])

    except ValueError as e:
        print(f"Backend: AI evaluation failed (ValueError): {e}")
        raise HTTPException(status_code=500, detail=f"AI evaluation error (Ollama): {e}")
    except Exception as e:
        print(f"Backend: General error during Ollama API call or DB operation for evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"AI evaluation error (Ollama): {e}")


# --- NEW: Endpoint for PDF Report Download ---
@app.get("/download-report/{session_id}")
async def download_report(session_id: int):
    print(f"Backend: /download-report/ endpoint called for session_id: {session_id}")

    # 1. Retrieve data from database
    query = interview_sessions.select().where(interview_sessions.c.id == session_id)
    session_data = await database.fetch_one(query)

    if not session_data:
        print(f"Backend: Error: Session ID {session_id} not found for PDF report.")
        raise HTTPException(status_code=404, detail="Interview session not found.")

    print(f"Backend: Retrieved session data for report generation. Filename: {session_data['original_filename']}")

    # 2. Prepare content for PDF
    pdf_content = io.BytesIO()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Interview Report", ln=True, align='C')
    pdf.ln(5)

    # Candidate Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Candidate: {session_data['parsed_resume_data'].get('name', 'N/A')}", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Job Role: {session_data['job_role']}", ln=True)
    pdf.cell(0, 5, f"Difficulty: {session_data['difficulty']}", ln=True)
    pdf.cell(0, 5, f"Date: {session_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    # Overall Summary (Optional: generate a summary from LLM here if needed, or from existing parsed_resume_data)
    # For now, let's include key resume data parts
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Resume Summary:", ln=True)
    pdf.set_font("Arial", '', 10)
    if 'summary' in session_data['parsed_resume_data']:
        pdf.multi_cell(0, 5, session_data['parsed_resume_data']['summary'])
    else:
        pdf.cell(0, 5, "No specific summary extracted.", ln=True)
    pdf.ln(5)

    # Interview Questions and Evaluations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Interview Questions & AI Evaluation:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.ln(5)

    if session_data['qa_evaluations']:
        for i, qa_eval in enumerate(session_data['qa_evaluations']):
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 6, f"{i+1}. Question: {qa_eval['question']}")
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 6, f"   Your Answer: {qa_eval['answer']}")
            pdf.set_text_color(0, 123, 255) # Blue color for score
            pdf.multi_cell(0, 6, f"   AI Score: {qa_eval['score']}/10")
            pdf.set_text_color(0, 0, 0) # Back to black
            pdf.set_font("Arial", 'I', 9) # Italic for feedback
            pdf.multi_cell(0, 6, f"   AI Feedback: {qa_eval['feedback']}")
            pdf.ln(5)
    else:
        pdf.cell(0, 5, "No questions answered or evaluated.", ln=True)

    # Final "Next Steps" or improvement suggestions (optional LLM call here or just static)
    # For a simple PDF, we can add a static section for now.
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Recommendations for Improvement:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, "Review the AI feedback for each question. Focus on areas where your score was lower. Practice articulating your experience with STAR method for behavioral questions and deepen your technical knowledge. Consider mock interviews for further improvement.")


    pdf.output(pdf_content)
    pdf_content.seek(0) # Go to the beginning of the stream

    print(f"Backend: PDF report generated for session {session_id}.")
    # 3. Return PDF as a downloadable response
    return Response(content=pdf_content.getvalue(), media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=interview_report_session_{session_id}.pdf"
    })