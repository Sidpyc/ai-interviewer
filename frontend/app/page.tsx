// frontend/app/page.tsx
'use client';

import { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

// Define UI states for the application flow
type AppState = 'initial' | 'uploading' | 'parsing_complete' | 'starting_interview' |
                'interview_active' | 'submitting_answer' | 'interview_complete' | 'generating_summary';

// Type for a message in the chat history
interface ChatMessage {
  type: 'ai' | 'user';
  content: string;
  turnNum?: number; // For questions/answers
  evaluation?: { // For user answers
    score: number;
    feedback: string;
  };
}

// Type for the full Q&A pair sent to the backend
interface QaPair {
  question: string;
  answer: string;
  score: number;
  feedback: string;
  turn_num: number;
}


export default function Home() {
  const [backendMessage, setBackendMessage] = useState<string>('Loading backend connection status...');
  const [error, setError] = useState<string | null>(null);

// eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_selectedFile, setSelectedFile] = useState<File | null>(null);
  const [appState, setAppState] = useState<AppState>('initial');
  const [currentStatus, setCurrentStatus] = useState<string>('');

  // --- STATE REFACTOR: Replace session_id with context state ---
  const [extractedResumeText, setExtractedResumeText] = useState<string | null>(null);
  const [parsedResumeData, setParsedResumeData] = useState<object | null>(null);

  const [jobRole, setJobRole] = useState<string>('');
  const [difficulty, setDifficulty] = useState<string>('medium');

  // States for conversational flow
  const [currentQuestion, setCurrentQuestion] = useState<string | null>(null);
  const [currentAnswer, setCurrentAnswer] = useState<string>('');
  const [turnNum, setTurnNum] = useState<number>(0);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  const [overallSummary, setOverallSummary] = useState<string | null>(null);
  
// eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_aiIntroMessage, setAiIntroMessage] = useState<string | null>(null);
// eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_aiClosingMessage, setAiClosingMessage] = useState<string | null>(null);

  const [textBeforeListening, setTextBeforeListening] = useState<string>('');

  // States for voice features
  const [ttsVoices, setTtsVoices] = useState<SpeechSynthesisVoice[]>([]);
  const {
    transcript,
    listening,
    browserSupportsSpeechRecognition,
    isMicrophoneAvailable,
    resetTranscript
  } = useSpeechRecognition();


  // Ref for scrolling to the bottom of the chat
  const messagesEndRef = useRef<HTMLDivElement>(null);


  // Backend Base URL
  const BACKEND_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_BASE_URL || 'http://localhost:8000';


  // Scroll to bottom whenever chatMessages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);


  // Handler to clear all application data and reset UI to initial state
  const handleClearAll = () => {
    setSelectedFile(null);
    const fileInput = document.getElementById('resumeFileInput') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
    setAppState('initial');
    setCurrentStatus('');
    setJobRole('');
    setDifficulty('medium');
    setChatMessages([]);
    setCurrentQuestion(null);
    setCurrentAnswer('');
    setTurnNum(0);
    setOverallSummary(null);
    setError(null);
    setAiIntroMessage(null);
    setAiClosingMessage(null);
    
    // --- STATE REFACTOR: Clear new context state ---
    setExtractedResumeText(null);
    setParsedResumeData(null);
    
    setTextBeforeListening('');
    
    resetTranscript();
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
  };


  // Effect hook to check backend connection status and initialize TTS
  useEffect(() => {
    const fetchMessage = async () => {
      try {
        const response = await fetch(`${BACKEND_BASE_URL}/api/message`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setBackendMessage(data.data);
      } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        setError(`Failed to load backend status: ${errorMessage}. Is the backend running at ${BACKEND_BASE_URL}?`);
        setBackendMessage('Error fetching backend status.');
      }
    };
    fetchMessage();

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.onvoiceschanged = () => {
        setTtsVoices(window.speechSynthesis.getVoices());
      };
      if (window.speechSynthesis.getVoices().length > 0) {
        setTtsVoices(window.speechSynthesis.getVoices());
      }
    }
    
    return () => {
      if (window.speechSynthesis) {
          window.speechSynthesis.cancel();
      }
    };

  }, [BACKEND_BASE_URL]);


  useEffect(() => {
    if (listening) {
      const separator = textBeforeListening && transcript ? ' ' : '';
      setCurrentAnswer(textBeforeListening + separator + transcript);
    }
  }, [transcript, textBeforeListening, listening]);


  // Handler for file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setCurrentStatus(`File selected: ${file.name}`);
      setError(null);
      // Automatically trigger upload after selection
      handleFileUpload(file);
    } else {
      setSelectedFile(null);
      setCurrentStatus('No file chosen or selection cancelled.');
    }
    event.target.value = ''; 
  };


  // Handler for initiating the file upload
  const handleFileUpload = async (fileToUpload: File) => {
    if (!fileToUpload) {
      setCurrentStatus('Please select a file first.');
      return;
    }

    handleClearAll(); 

    setAppState('uploading');
    setCurrentStatus('Processing resume with AI (this may take a few minutes for initial model loading)...');
    setError(null);

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await fetch(`${BACKEND_BASE_URL}/upload-resume/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Upload failed with status: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
      }

      const result = await response.json();
      
      // --- STATE REFACTOR: Store context from backend ---
      setExtractedResumeText(result.extracted_resume_text);
      setParsedResumeData(result.parsed_resume_data);
      
      setCurrentStatus('Resume processed successfully! Ready to start interview.');
      setAppState('parsing_complete');

    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCurrentStatus(`Upload failed: ${errorMessage}`);
      setError(errorMessage);
      setAppState('initial');
    }
  };


  // Handler for starting the conversational interview
  const handleStartInterview = async () => {
    // --- STATE REFACTOR: Check for context instead of session ID ---
    if (appState !== 'parsing_complete' || !extractedResumeText) {
      setCurrentStatus('Please upload and process a resume first.');
      return;
    }
    if (!jobRole.trim()) {
      setCurrentStatus('Please enter a job role.');
      return;
    }

    setAppState('starting_interview');
    setCurrentStatus('Starting interview and generating initial greeting...');
    setError(null);
    setChatMessages([]);
    setOverallSummary(null);
    setAiIntroMessage(null);
    setAiClosingMessage(null);

    try {
      const response = await fetch(`${BACKEND_BASE_URL}/start-interview/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // --- API REFACTOR: Send context in body ---
        body: JSON.stringify({
          extracted_resume_text: extractedResumeText,
          job_role: jobRole,
          difficulty: difficulty,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to start interview: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
      }

      const result = await response.json();
      const aiInitialMessage = result.initial_ai_message;
      const firstQuestion = result.first_question;
      const totalQuestions = result.total_questions;
      const turnNumber = result.turn_num;

      // The first message is the intro, not a question to be answered
      // The second message is the first question
      setChatMessages([
        { type: 'ai', content: aiInitialMessage },
        { type: 'ai', content: firstQuestion, turnNum: turnNumber }
      ]);

      setAiIntroMessage(aiInitialMessage);
      setCurrentQuestion(firstQuestion);
      setTurnNum(turnNumber);
      setCurrentAnswer('');
      setCurrentStatus(`Interview active: Question ${turnNumber} of ${totalQuestions}.`); 
      setAppState('interview_active');

    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCurrentStatus(`Failed to start interview: ${errorMessage}`);
      setError(errorMessage);
      setAppState('parsing_complete');
    }
  };


  const handleMicClick = () => {
    if (!browserSupportsSpeechRecognition) {
      const msg = "Browser does not support Speech Recognition.";
      setCurrentStatus(msg);
      setError(msg);
      return;
    }
    if (!isMicrophoneAvailable) {
      const msg = "Microphone not available. Please check permissions.";
      setCurrentStatus(msg);
      setError(msg);
      return;
    }

    if (listening) {
        SpeechRecognition.stopListening();
        setCurrentStatus('Speech recognition stopped.');
    } else {
        setTextBeforeListening(currentAnswer); 
        resetTranscript();
        SpeechRecognition.startListening({ continuous: true, language: 'en-US' });
        setCurrentStatus('Listening...');
        setError(null);
    }
  };


  const handleSpeakerClick = (textToSpeak: string) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      setCurrentStatus("Text-to-speech not supported in your browser.");
      setError("Voice output not available.");
      return;
    }

    window.speechSynthesis.cancel(); 

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.lang = 'en-US';

    if (ttsVoices.length > 0) {
        utterance.voice = ttsVoices.find(voice => voice.lang === 'en-US' && voice.name.includes('Google') && voice.name.includes('US')) || ttsVoices[0];
    }
    
    window.speechSynthesis.speak(utterance);
  };


  // Handler for submitting an answer and getting the next question
  const handleSubmitAnswer = async () => {
    // --- STATE REFACTOR: Check for context ---
    if (!currentQuestion || currentAnswer.trim() === '' || !extractedResumeText) {
      setCurrentStatus('Please provide an answer before submitting.');
      return;
    }

    if (listening) {
        SpeechRecognition.stopListening();
    }
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }

    setAppState('submitting_answer');
    setCurrentStatus(`Submitting answer for Question ${turnNum} and getting next question...`);
    setError(null);

    const answeredQuestionText = currentQuestion;
    const submittedAnswerText = currentAnswer;
    const currentTurnNumber = turnNum;

    setChatMessages(prevMessages => [...prevMessages, {
      type: 'user',
      content: submittedAnswerText,
      turnNum: currentTurnNumber
    }]);

    setCurrentAnswer('');
    setTextBeforeListening('');

    try {
      // --- API REFACTOR: Construct full history for context ---
      const qa_history: QaPair[] = chatMessages
        .filter(msg => msg.type === 'user' && msg.evaluation)
        .map(msg => {
          const questionMsg = chatMessages.find(q => q.type === 'ai' && q.turnNum === msg.turnNum);
          return {
            question: questionMsg?.content || '',
            answer: msg.content,
            score: msg.evaluation!.score,
            feedback: msg.evaluation!.feedback,
            turn_num: msg.turnNum!,
          };
        });

      const response = await fetch(`${BACKEND_BASE_URL}/submit-answer/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        // --- API REFACTOR: Send full context ---
        body: JSON.stringify({
          extracted_resume_text: extractedResumeText,
          job_role: jobRole,
          difficulty: difficulty,
          qa_history: qa_history,
          current_question: answeredQuestionText,
          candidate_answer: submittedAnswerText,
          turn_num: currentTurnNumber,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to submit answer: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
      }

      const result = await response.json();
      
      setChatMessages(prevMessages => 
        prevMessages.map(msg => 
            msg.type === 'user' && msg.turnNum === currentTurnNumber 
                ? { ...msg, evaluation: result.answered_question_evaluation }
                : msg
        )
      );

      if (result.interview_complete) {
        setCurrentQuestion(null);
        setAiClosingMessage(result.ai_closing_message);
        
        if (result.ai_closing_message) {
          setChatMessages(prevMessages => [...prevMessages, { type: 'ai', content: result.ai_closing_message }]);
        }

        setCurrentStatus('Interview complete! Generating overall summary...');
        await fetchOverallSummary();
        setAppState('interview_complete');
      } else {
        const nextQuestionText = result.next_question;
        const nextTurnNumber = result.turn_num;
        setCurrentQuestion(nextQuestionText);
        setTurnNum(nextTurnNumber);
        
        setChatMessages(prevMessages => [...prevMessages, { type: 'ai', content: nextQuestionText, turnNum: nextTurnNumber }]);

        setCurrentStatus(`Question ${nextTurnNumber}:`);
        setAppState('interview_active');
      }

    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCurrentStatus(`Failed to submit answer: ${errorMessage}`);
      setError(errorMessage);
      setAppState('interview_active');
    }
  };


  // Handler for fetching overall summary from backend
  const fetchOverallSummary = async () => {
    // --- STATE REFACTOR: Check for context ---
    if (!parsedResumeData) {
        setOverallSummary('Error: No resume data available to generate summary.');
        return;
    }
    setAppState('generating_summary');
    setCurrentStatus('Generating final interview summary (this may take a moment)...');
    try {
        // --- API REFACTOR: Construct full history for summary ---
        const qa_evaluations: QaPair[] = chatMessages
          .filter(msg => msg.type === 'user' && msg.evaluation)
          .map(msg => {
            const questionMsg = chatMessages.find(q => q.type === 'ai' && q.turnNum === msg.turnNum);
            return {
              question: questionMsg?.content || '',
              answer: msg.content,
              score: msg.evaluation!.score,
              feedback: msg.evaluation!.feedback,
              turn_num: msg.turnNum!,
            };
          });

        // --- API REFACTOR: Use POST and send body ---
        const response = await fetch(`${BACKEND_BASE_URL}/generate-summary/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', },
            body: JSON.stringify({
              parsed_resume_data: parsedResumeData,
              job_role: jobRole,
              difficulty: difficulty,
              qa_evaluations: qa_evaluations,
            }),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Summary generation failed: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
        }
        const result = await response.json();
        setOverallSummary(result.summary);
        setCurrentStatus('Overall summary complete! Scroll down to view.');
        setAppState('interview_complete');

    } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        setOverallSummary(`Failed to generate overall summary: ${errorMessage}`);
        setCurrentStatus(`Summary generation failed: ${errorMessage}`);
        setError(errorMessage);
        setAppState('interview_complete');
    }
  };


  return (
    <main className="min-h-screen bg-gray-950 text-gray-50 p-4">
      <div className="max-w-3xl mx-auto py-8">
        <h1 className="text-4xl font-extrabold text-center mb-6 text-blue-400">AI Interviewer</h1>
        <p className="text-center text-sm mb-8 text-gray-400">Backend Connection Status: <strong className="text-blue-500">{backendMessage}</strong></p>

        <hr className="my-8 border-gray-700" />

        {/* Overall Status Message & Error Display */}
        {(currentStatus || error) && (
          <div className={`p-4 rounded-md mb-6 font-semibold shadow-md ${
            error
              ? 'bg-red-800 text-red-100 border border-red-600'
              : (currentStatus.includes('successfully') || currentStatus.includes('generated') || currentStatus.includes('complete'))
                ? 'bg-green-800 text-green-100 border border-green-600'
                : 'bg-yellow-800 text-yellow-100 border border-yellow-600'
          }`}>
            {currentStatus} {error && `: ${error}`}
          </div>
        )}

        {/* Step 1: Upload Resume Section - Visible only in 'initial' state */}
        {appState === 'initial' && (
          <div className="bg-gray-800 p-8 rounded-lg shadow-xl mb-8 border border-gray-700">
            <h2 className="text-2xl font-bold text-blue-400 mb-6">Step 1: Upload Resume for AI Analysis</h2>
            <div className="mb-4">
              <label htmlFor="resumeFileInput" className="block text-sm font-medium text-gray-300 mb-2">
                Select your resume (PDF, DOCX):
              </label>
              <input
                type="file"
                id="resumeFileInput"
                accept=".pdf,.docx"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-300
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-full file:border-0
                         file:text-sm file:font-semibold
                         file:bg-blue-600 file:text-white
                         hover:file:bg-blue-700 dark:file:bg-blue-700 dark:file:hover:bg-blue-800
                         dark:text-gray-200 cursor-pointer"
              />
            </div>
            {/* Removed manual upload button to streamline flow */}
          </div>
        )}

        {/* Loading Spinner for Resume Parsing (Upload phase) */}
        {appState === 'uploading' && (
          <div className="flex flex-col items-center justify-center p-8 rounded-lg shadow-lg mb-8
                          bg-gray-800 border border-blue-700">
            <div className="spinner mb-4 w-12 h-12"></div>
            <p className="text-lg font-medium text-blue-400">{currentStatus}</p>
          </div>
        )}

        {/* Step 2: Job Role & Difficulty Selection + Start Interview Button */}
        {(appState === 'parsing_complete' || appState === 'starting_interview') && (
          <div className="bg-gray-800 p-8 rounded-lg shadow-xl mb-8 border border-gray-700">
            <h2 className="text-2xl font-bold text-green-400 mb-6">Step 2: Configure & Start Interview</h2>
            <div className="mb-4">
              <label htmlFor="jobRole" className="block text-base font-medium text-gray-300 mb-2">Job Role:</label>
              <input
                type="text"
                id="jobRole"
                value={jobRole}
                onChange={(e) => setJobRole(e.target.value)}
                placeholder="e.g., Software Engineer, AI Specialist"
                disabled={appState === 'starting_interview'}
                className="w-full p-3 border border-gray-600 rounded-md bg-gray-700 text-gray-100 placeholder-gray-400
                           focus:ring-green-500 focus:border-green-500"
              />
            </div>
            <div className="mb-6">
              <label htmlFor="difficulty" className="block text-base font-medium text-gray-300 mb-2">Difficulty:</label>
              <select
                id="difficulty"
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                disabled={appState === 'starting_interview'}
                className="w-full p-3 border border-gray-600 rounded-md bg-gray-700 text-gray-100
                           focus:ring-green-500 focus:border-green-500 h-11 text-base"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
            <div className="flex space-x-4">
              <button
                onClick={handleStartInterview}
                disabled={!jobRole.trim() || appState === 'starting_interview'}
                className="flex-1 px-6 py-3 rounded-lg font-bold text-white transition-colors duration-200
                           bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed text-lg"
              >
                {appState === 'starting_interview' ? 'Starting...' : 'Start Interview'}
              </button>
              <button 
                onClick={handleClearAll}
                className="px-6 py-3 rounded-lg font-bold text-white transition-colors duration-200
                           bg-red-600 hover:bg-red-700"
              >
                Start Over
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Conversational Interview - Visible when interview is active or completing */}
        {(appState === 'interview_active' || appState === 'submitting_answer' || appState === 'interview_complete' || appState === 'generating_summary') && (
          <div className="bg-gray-800 p-8 rounded-lg shadow-xl mb-8 border border-gray-700">
            <h2 className="text-2xl font-bold text-blue-400 mb-6">Interview In Progress</h2>
            
            {/* Chat Messages Container */}
            <div className="h-96 overflow-y-auto p-4 mb-6 bg-gray-900 rounded-lg border border-gray-700 flex flex-col space-y-4">
              {chatMessages.map((message, index) => (
                <div 
                  key={index} 
                  className={`flex ${message.type === 'ai' ? 'justify-start' : 'justify-end'}`}
                >
                  <div className={`p-3 rounded-lg max-w-[80%] shadow ${
                    message.type === 'ai'
                      ? 'bg-gray-700 text-gray-200'
                      : 'bg-blue-600 text-white'
                  }`}>
                    <p className="font-semibold text-blue-300 mb-1">
                      {message.type === 'ai' ? 'AI Interviewer:' : 'Your Answer:'}
                    </p>
                    {message.type === 'ai' && message.turnNum && (
                      <p className="text-sm leading-relaxed font-bold mb-1">Question {message.turnNum}:</p>
                    )}
                    <p className="text-sm leading-relaxed" style={{ whiteSpace: 'pre-wrap' }}>
                      {message.content}
                    </p>
                    {message.type === 'user' && message.evaluation && (
                      <div className="mt-2 p-2 rounded-md bg-gray-800 border border-blue-500 text-xs text-gray-300 shadow-inner">
                        <p className="font-semibold text-blue-400">Score: {message.evaluation.score}/10</p>
                        <p>{message.evaluation.feedback}</p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} /> {/* For auto-scrolling */}
            </div>

            {/* Current Answer Input Area - Only visible if interview is active */}
            {appState === 'interview_active' && currentQuestion && (
              <div className="mb-6">
                <div className="flex space-x-2 mb-2">
                  <button
                    onClick={handleMicClick}
                    className={`px-4 py-2 rounded-md font-semibold text-white transition-colors duration-200 text-sm ${
                      listening ? 'bg-red-600 hover:bg-red-700' : 'bg-purple-600 hover:bg-purple-700'
                    }`}
                  >
                    {listening ? 'Stop Listening' : 'Use Mic'}
                  </button>
                  <button
                    onClick={() => handleSpeakerClick(currentQuestion)}
                    className="px-4 py-2 rounded-md font-semibold text-white transition-colors duration-200 bg-indigo-600 hover:bg-indigo-700 text-sm"
                  >
                    Read Question
                  </button>
                </div>
                <textarea
                  value={currentAnswer}
                  onChange={(e) => setCurrentAnswer(e.target.value)}
                  rows={4}
                  placeholder="Type your answer here or use the mic..."
                  disabled={listening}
                  className="w-full p-3 border border-gray-600 rounded-lg bg-gray-700 text-gray-100 placeholder-gray-400
                             focus:ring-blue-500 focus:border-blue-500 resize-y focus:outline-none"
                />
                <button
                  onClick={handleSubmitAnswer}
                  disabled={currentAnswer.trim() === ''}
                  className="w-full mt-2 px-6 py-3 rounded-lg font-bold text-white transition-colors duration-200
                             bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed text-lg"
                >
                  Submit Answer
                </button>
              </div>
            )}

            {/* Loading spinner while submitting answer */}
            {appState === 'submitting_answer' && (
              <div className="flex flex-col items-center justify-center p-8">
                  <div className="spinner mb-4 w-12 h-12"></div>
                  <p className="text-lg font-medium text-blue-400">{currentStatus}</p>
              </div>
            )}

            {/* Final Summary Section */}
            {(appState === 'interview_complete' || appState === 'generating_summary') && (
              <div className="mt-8 pt-6 border-t border-gray-700">
                <h3 className="text-xl font-semibold text-gray-300 mb-4">Overall Interview Summary:</h3>
                {overallSummary ? (
                  <div className="bg-gray-700 p-4 rounded-lg border border-blue-600 text-gray-200 whitespace-pre-wrap break-words max-h-96 overflow-y-auto shadow-inner">
                    {overallSummary}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center p-8">
                    <div className="spinner mb-4 w-12 h-12"></div>
                    <p className="text-lg font-medium text-blue-400">{currentStatus}</p>
                  </div>
                )}
              </div>
            )}
            
            <div className="flex justify-center space-x-4 mt-8 flex-wrap">
              {appState === 'interview_complete' && (
                  <button 
                      onClick={handleClearAll}
                      className="px-6 py-3 rounded-lg font-bold text-white transition-colors duration-200
                                 bg-red-600 hover:bg-red-700"
                  >
                      Start New Interview
                  </button>
              )}
              {appState === 'interview_active' && (
                  <button
                      onClick={() => {
                          if (window.confirm("Are you sure you want to end the interview early?")) {
                              setCurrentQuestion(null);
                              setAppState('interview_complete');
                              setCurrentStatus('Interview ended early. Generating summary...');
                              fetchOverallSummary();
                          }
                      }}
                      className="px-6 py-3 rounded-lg font-bold text-gray-900 transition-colors duration-200
                                 bg-yellow-500 hover:bg-yellow-600"
                  >
                      End Interview Early
                  </button>
              )}
            </div>
          </div>
        )}

      </div>
    </main>
  );
}