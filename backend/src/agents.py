from .config import MAX_QUESTIONS, TECHNICAL_QUESTIONS_COUNT
from .llm import _call_llm, _repair_llm_json


async def _generate_next_question_llm(
    resume_text: str,
    job_role: str,
    difficulty: str,
    qa_history: list[dict],
    current_turn_num: int
) -> tuple[str | None, str | None]:
    
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
        if current_turn_num == 1:
            return "Please tell me about yourself.", None
        elif current_turn_num <= (1 + TECHNICAL_QUESTIONS_COUNT):
             return "Could you describe your approach to optimizing code performance, or tell me about a technical project you worked on?", None
        else:
             return "Tell me about a time you had to overcome a significant challenge at work, how did you handle it?", None

    return generated_question.strip(), None
