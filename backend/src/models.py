from pydantic import BaseModel, Field


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
    extracted_resume_text: str
    job_role: str
    difficulty: str
    qa_history: list[QuestionAnswerPair]
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
