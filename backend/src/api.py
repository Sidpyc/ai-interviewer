import io
import os
import tempfile

from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from docling.document_converter import DocumentConverter

from .config import MAX_QUESTIONS
from .llm import _call_llm, _repair_llm_json, init_llm_client
from .models import (
    AnswerEvaluation,
    GenerateSummaryRequest,
    OverallSummaryResponse,
    QuestionAnswerPair,
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    UploadResumeResponse,
)
from .agents import _generate_next_question_llm


docling_converter = DocumentConverter()


def create_app() -> FastAPI:
    app = FastAPI()

    origins = [
        "http://localhost",
        "http://localhost:3000",
        "https://ai-interviewer-beta-six.vercel.app",
        "https://ai-interviewer-1-lj0z.onrender.com",
        "*"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        init_llm_client()

    @app.get("/")
    async def read_root():
        return {"message": "Hello from FastAPI Backend!"}

    @app.get("/api/message")
    async def send_message():
        return {"data": "Backend is Running"}

    @app.post("/upload-resume/", response_model=UploadResumeResponse)
    async def upload_resume(file: UploadFile = File(...)):
        try:
            fd, temp_file_path = tempfile.mkstemp(suffix=f".{file.filename.split('.')[-1]}")
            
            try:
                with os.fdopen(fd, 'wb') as tmp:
                    content = await file.read()
                    tmp.write(content)
                
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

        return UploadResumeResponse(
            original_filename=file.filename,
            extracted_resume_text=extracted_resume_text,
            parsed_resume_data=parsed_resume_data
        )


    @app.post("/start-interview/", response_model=StartInterviewResponse)
    async def start_interview(request: StartInterviewRequest):
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
        resume_text_context = request.extracted_resume_text
        current_job_role = request.job_role
        current_difficulty = request.difficulty
        qa_history_from_client = [item.model_dump() for item in request.qa_history]

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

        updated_qa_history = qa_history_from_client + [{
            "question": request.current_question,
            "answer": request.candidate_answer,
            "score": current_evaluation.score,
            "feedback": current_evaluation.feedback,
            "turn_num": request.turn_num
        }]

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
            - Resume Summary: {resume_summary}r
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

    return app
