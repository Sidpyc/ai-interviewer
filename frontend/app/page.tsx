// frontend/app/page.tsx
'use client';

import { useState, useEffect } from 'react';

// Define UI states for the application flow
type AppState = 'initial' | 'uploading' | 'parsing_complete' | 'generating_questions' | 'interview_active' | 'evaluating_answers' | 'evaluation_complete';

// Type for an individual interview question, including user's answer and AI evaluation
interface InterviewQuestion {
  question: string;
  answer: string; // User's typed answer
  evaluation?: { // Optional AI evaluation data
    score: number;
    feedback: string;
  };
}

export default function Home() {
  // State for backend connection status
  const [backendMessage, setBackendMessage] = useState<string>('Loading backend connection status...');
  // State for general error messages
  const [error, setError] = useState<string | null>(null);

  // State for the currently selected file
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // Controls the current step/section displayed in the UI
  const [appState, setAppState] = useState<AppState>('initial');
  // Displays detailed status messages to the user
  const [currentStatus, setCurrentStatus] = useState<string>('');

  // State to store the current interview session ID from the backend
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);

  // States for job role and difficulty for question generation
  const [jobRole, setJobRole] = useState<string>('');
  const [difficulty, setDifficulty] = useState<string>('medium');
  // Stores the generated interview questions along with user's answers and AI evaluation
  const [interviewQuestions, setInterviewQuestions] = useState<InterviewQuestion[]>([]);

  // State for the AI-generated overall summary (replaces PDF download for now)
  const [overallSummary, setOverallSummary] = useState<string | null>(null);


  // Effect hook to check backend connection status on component mount
  useEffect(() => {
    const fetchMessage = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/message');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setBackendMessage(data.data);
      } catch (e: unknown) { // FIX: Changed e: any to e: unknown
        // For 'unknown' type, you often need to check the type before accessing properties
        const errorMessage = e instanceof Error ? e.message : String(e);
        setError(`Failed to load backend status: ${errorMessage}. Is the backend running at http://localhost:8000?`);
        setBackendMessage('Error fetching backend status.');
      }
    };
    fetchMessage();
  }, []); // Empty dependency array means this runs once on mount


  // Handler for when a file is selected via the input
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
      // Reset all relevant states for a fresh start with new file
      setCurrentStatus('');
      setError(null);
      setAppState('initial');
      setInterviewQuestions([]);
      setJobRole('');
      setDifficulty('medium');
      setCurrentSessionId(null); // Clear session ID on new file selection
      setOverallSummary(null); // Clear overall summary
    } else {
      setSelectedFile(null);
    }
    // Crucial for allowing re-selection of the same file in some browsers
    event.target.value = ''; 
  };


  // Handler for initiating the file upload to the backend
  const handleFileUpload = async () => {
    if (!selectedFile) {
      setCurrentStatus('Please select a file first.');
      return;
    }

    setAppState('uploading'); // Transition UI to 'uploading' state
    setCurrentStatus('Processing resume with AI (this may take a few minutes for initial model loading)...'); // Update status message
    setError(null);
    setInterviewQuestions([]); // Clear any previous interview data
    setCurrentSessionId(null); // Ensure no old session ID is used
    setOverallSummary(null); // Clear overall summary

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/upload-resume/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Upload failed with status: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
      }

      const result = await response.json();
      setCurrentSessionId(result.session_id); // Store the new session ID
      setCurrentStatus('Resume processed successfully! Now, let\'s generate questions.');
      setAppState('parsing_complete'); // Transition UI to 'parsing_complete' state

    } catch (e: unknown) { // FIX: Changed e: any to e: unknown
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCurrentStatus(`Upload failed: ${errorMessage}`);
      setError(errorMessage);
      setAppState('initial'); // Revert to 'initial' state on error
    }
  };


  // Handler for generating interview questions from the backend
  const handleGenerateQuestions = async () => {
    if (appState !== 'parsing_complete' || currentSessionId === null) { // Ensure a resume has been parsed and session exists
      setCurrentStatus('Please upload and parse a resume first to start a session.');
      return;
    }
    if (!jobRole.trim()) { // Ensure job role is provided
        setCurrentStatus('Please enter a job role.');
        return;
    }

    setAppState('generating_questions'); // Transition UI to 'generating_questions' state
    setCurrentStatus('Generating personalized interview questions...'); // Update status message
    setError(null);
    setInterviewQuestions([]); // Clear any old questions
    setOverallSummary(null); // Clear overall summary

    try {
      const response = await fetch('http://localhost:8000/generate-questions/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: currentSessionId, // Send the session ID
          job_role: jobRole,
          difficulty: difficulty,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Question generation failed: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
      }

      const result = await response.json();
      // Initialize questions with empty answers for user input
      const initialQuestions: InterviewQuestion[] = (result.questions || []).map((q: string) => ({
        question: q,
        answer: '' // Each question starts with an empty answer
      }));
      setInterviewQuestions(initialQuestions);
      setCurrentStatus('10 questions generated! Please answer them below.'); // Update status
      setAppState('interview_active'); // Transition UI to 'interview_active' state

    } catch (e: unknown) { // FIX: Changed e: any to e: unknown
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCurrentStatus(`Question generation failed: ${errorMessage}`);
      setError(errorMessage);
      setAppState('parsing_complete'); // Revert to 'parsing_complete' state on error
    }
  };


  // Handler for updating a user's answer in the state as they type
  const handleAnswerChange = (index: number, value: string) => {
    setInterviewQuestions(prevQuestions =>
      prevQuestions.map((q, i) =>
        i === index ? { ...q, answer: value } : q // Update only the specific question's answer
      )
    );
  };


  // Handler for submitting all answers for AI evaluation
  const handleSubmitAnswers = async () => {
    // Check if any answer box is still empty
    if (interviewQuestions.some(q => q.answer.trim() === '')) {
      setCurrentStatus('Please answer all questions before submitting.');
      return;
    }
    if (currentSessionId === null) {
        setCurrentStatus('Error: No active interview session. Please start over.');
        return;
    }

    setAppState('evaluating_answers'); // Transition UI to 'evaluating_answers' state
    setCurrentStatus('AI is evaluating your answers and generating feedback...'); // Update status
    setError(null);
    setOverallSummary(null); // Clear previous summary

    try {
      // Make API call to the backend evaluation endpoint
      const response = await fetch('http://localhost:8000/evaluate-answers/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: currentSessionId, // Send the session ID for context and storage
          questions_with_answers: interviewQuestions.map(q => ({
            question: q.question,
            answer: q.answer
          }))
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Evaluation failed with status: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
      }

      const evaluationResult = await response.json(); // Expected: { evaluations: [{ score, feedback }] }
      setCurrentStatus('Evaluation complete! Generating overall summary...'); // Status update for summary
      
      // Update the interviewQuestions state with the received evaluation results
      setInterviewQuestions(prevQuestions =>
        prevQuestions.map((q, index) => ({
          ...q,
          evaluation: evaluationResult.evaluations[index] || { score: 0, feedback: 'No feedback.' } // Assign evaluation data
        }))
      );
      // Trigger overall summary generation after individual evaluation
      await fetchOverallSummary(); 
      // After fetching summary, set final state
      setAppState('evaluation_complete'); 

    } catch (e: unknown) { // FIX: Changed e: any to e: unknown
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCurrentStatus(`Evaluation failed: ${errorMessage}`);
      setError(errorMessage);
      setAppState('interview_active'); // Revert to 'interview_active' state on error
    }
  };


  // Handler for fetching overall summary from backend
  const fetchOverallSummary = async () => {
    if (currentSessionId === null) {
        setOverallSummary('Error: No session ID to generate summary.');
        return;
    }
    try {
        const response = await fetch(`http://localhost:8000/generate-summary/${currentSessionId}`, {
            method: 'GET',
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Summary generation failed: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
        }
        const result = await response.json();
        setOverallSummary(result.summary);
        setCurrentStatus('Evaluation and overall summary complete! Scroll down to view the summary.');
    } catch (e: unknown) { // FIX: Changed e: any to e: unknown
      const errorMessage = e instanceof Error ? e.message : String(e);
      setOverallSummary(`Failed to generate overall summary: ${errorMessage}`);
      setError(errorMessage);
    }
  };


  // Handler to clear all application data and reset UI to initial state
  const handleClearAll = () => {
    setSelectedFile(null);
    const fileInput = document.getElementById('resumeFileInput') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = ''; // Ensure the file input is cleared
    }
    setAppState('initial');
    setCurrentStatus('');
    setJobRole('');
    setDifficulty('medium');
    setInterviewQuestions([]); // Clear all questions and answers
    setError(null);
    setCurrentSessionId(null); // Clear the session ID
    setOverallSummary(null); // Clear the overall summary
  };


  return (
    <main style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '800px', margin: '0 auto' }}>
      <h1 style={{ color: '#333' }}>AI Interviewer - Frontend</h1>
      <p style={{ color: '#555' }}>Backend Connection Status: <strong>{backendMessage}</strong></p>

      <hr style={{ margin: '30px 0', borderColor: '#eee' }} />

      {/* Overall Status Message & Error Display */}
      {(currentStatus || error) && (
        <div style={{
          padding: '10px 15px',
          backgroundColor: error ? '#ffebee' : (currentStatus.includes('successfully') || currentStatus.includes('generated') || currentStatus.includes('complete')) ? '#e8f5e9' : '#fff3e0',
          color: error ? '#c62828' : (currentStatus.includes('successfully') || currentStatus.includes('generated') || currentStatus.includes('complete')) ? '#2e7d32' : '#ff6f00',
          border: `1px solid ${error ? '#ef9a9a' : (currentStatus.includes('successfully') || currentStatus.includes('generated') || currentStatus.includes('complete')) ? '#a5d6a7' : '#ffb74d'}`,
          borderRadius: '5px',
          marginBottom: '20px',
          fontWeight: 'bold'
        }}>
          {currentStatus}
        </div>
      )}

      {/* Step 1: Upload Resume Section - Visible only in 'initial' state */}
      {appState === 'initial' && (
        <section style={{ marginBottom: '40px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
          <h2 style={{ color: '#0070f3' }}>Step 1: Upload Resume for AI Analysis</h2>
          <div style={{ marginBottom: '15px' }}>
            <input
              type="file"
              id="resumeFileInput"
              accept=".pdf,.docx"
              onChange={handleFileChange}
              style={{ border: '1px solid #ccc', padding: '10px', borderRadius: '5px', width: '100%', boxSizing: 'border-box' }}
            />
          </div>
          <p style={{ fontSize: '0.9em', color: '#555', marginBottom: '15px' }}>
            Selected File: <strong>{selectedFile ? selectedFile.name : 'No file chosen'}</strong>
          </p>
          <button
            onClick={handleFileUpload}
            disabled={!selectedFile}
            style={{
              padding: '10px 20px',
              backgroundColor: selectedFile ? '#0070f3' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: selectedFile ? 'pointer' : 'not-allowed',
              fontSize: '16px',
              marginRight: '10px'
            }}
          >
            Upload & Process Resume
          </button>
          <button
            onClick={handleClearAll}
            style={{
              padding: '10px 20px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Clear All
          </button>
        </section>
      )}

      {/* Loading Spinner for Resume Parsing (Upload phase) */}
      {appState === 'uploading' && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '50px', border: '1px solid #0070f3', borderRadius: '8px',
          flexDirection: 'column', marginBottom: '40px', backgroundColor: '#e6f7ff'
        }}>
          <div className="spinner" style={{
              border: '4px solid rgba(0, 0, 0, 0.1)',
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              borderLeftColor: '#0070f3',
              animation: 'spin 1s ease infinite',
              marginBottom: '15px'
          }}></div>
          <p style={{ fontSize: '1.2em', color: '#0070f3' }}>{currentStatus}</p>
          {/* Inline style for keyframe animation */}
          <style jsx>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}

      {/* Step 2: Question Generation Section - Visible after parsing_complete and all subsequent states */}
      {(appState === 'parsing_complete' || appState === 'generating_questions' || appState === 'interview_active' || appState === 'evaluating_answers' || appState === 'evaluation_complete') && (
        <section style={{ marginBottom: '40px', border: '1px solid #28a745', padding: '20px', borderRadius: '8px', backgroundColor: '#f0fff0' }}>
          <h2 style={{ color: '#28a745' }}>Step 2: Generate Interview Questions</h2>
          <div style={{ marginBottom: '15px' }}>
            <label htmlFor="jobRole" style={{ display: 'block', marginBottom: '5px' }}>Job Role:</label>
            <input
              type="text"
              id="jobRole"
              value={jobRole}
              onChange={(e) => setJobRole(e.target.value)}
              placeholder="e.g., Software Engineer, AI Specialist"
              disabled={appState === 'generating_questions' || appState === 'interview_active' || appState === 'evaluating_answers' || appState === 'evaluation_complete'}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '5px',
                border: '1px solid #ccc',
                boxSizing: 'border-box'
              }}
            />
          </div>
          <div style={{ marginBottom: '20px' }}>
            <label htmlFor="difficulty" style={{ display: 'block', marginBottom: '5px' }}>Difficulty:</label>
            <select
              id="difficulty"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              disabled={appState === 'generating_questions' || appState === 'interview_active' || appState === 'evaluating_answers' || appState === 'evaluation_complete'}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '5px',
                border: '1px solid #ccc',
                boxSizing: 'border-box',
                height: '40px'
              }}
            >
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>
          <button
            onClick={handleGenerateQuestions}
            disabled={!jobRole.trim() || appState === 'generating_questions' || appState === 'interview_active' || appState === 'evaluating_answers' || appState === 'evaluation_complete'}
            style={{
              padding: '10px 20px',
              backgroundColor: (jobRole.trim() && appState !== 'generating_questions' && appState !== 'interview_active' && appState !== 'evaluating_answers' && appState !== 'evaluation_complete') ? '#28a745' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '5px', 
              cursor: (jobRole.trim() && appState !== 'generating_questions' && appState !== 'interview_active' && appState !== 'evaluating_answers' && appState !== 'evaluation_complete') ? 'pointer' : 'not-allowed',
              fontSize: '16px',
              marginRight: '10px'
            }}
          >
            {appState === 'generating_questions' ? 'Generating...' : 'Generate Questions'}
          </button>
          <button
            onClick={handleClearAll}
            style={{
              padding: '10px 20px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Start Over
          </button>
        </section>
      )}

      {/* Loading Spinner for Question Generation */}
      {appState === 'generating_questions' && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '50px', border: '1px solid #28a745', borderRadius: '8px',
          flexDirection: 'column', marginBottom: '40px', backgroundColor: '#f0fff0'
        }}>
          <div className="spinner" style={{
              border: '4px solid rgba(0, 0, 0, 0.1)',
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              borderLeftColor: '#28a745',
              animation: 'spin 1s ease infinite',
              marginBottom: '15px'
          }}></div>
          <p style={{ fontSize: '1.2em', color: '#28a745' }}>{currentStatus}</p>
          <style jsx>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}

      {/* Step 3: Interview Questions and Answer Boxes - Visible when interview is active or evaluation complete */}
      {(appState === 'interview_active' || appState === 'evaluation_complete' || appState === 'evaluating_answers') && interviewQuestions.length > 0 && (
        <section style={{ marginBottom: '40px', border: '1px solid #0056b3', padding: '20px', borderRadius: '8px', backgroundColor: '#e0f7fa' }}>
          <h2 style={{ color: '#0056b3' }}>Step 3: Answer Interview Questions</h2>
          <p style={{marginBottom: '20px', fontSize: '0.9em', color: '#333'}}>
            Answer each question in the box below. Click "Submit Answers" when you are done.
          </p>
          {interviewQuestions.map((item, index) => (
            <div key={index} style={{ marginBottom: '25px', padding: '15px', border: '1px solid #b3e5fc', borderRadius: '5px', backgroundColor: 'white' }}>
              <p style={{ fontWeight: 'bold', marginBottom: '10px' }}>{index + 1}. {item.question}</p>
              <textarea
                value={item.answer}
                onChange={(e) => handleAnswerChange(index, e.target.value)}
                rows={5}
                placeholder="Type your answer here..."
                disabled={appState === 'evaluating_answers' || appState === 'evaluation_complete'} // Disable while evaluating or after complete
                style={{
                  width: '100%',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #ccc',
                  boxSizing: 'border-box',
                  resize: 'vertical'
                }}
              />
              {/* AI evaluation for individual questions */}
              {item.evaluation && (
                <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#f0f8ff', borderLeft: '3px solid #007bff', fontSize: '0.9em' }}>
                  <p style={{ fontWeight: 'bold', color: '#007bff' }}>AI Score: {item.evaluation.score}/10</p>
                  <p style={{ color: '#333' }}>AI Feedback: {item.evaluation.feedback}</p>
                </div>
              )}
            </div>
          ))}
          {/* Action buttons after answering/evaluation */}
          <div style={{ display: 'flex', justifyContent: 'flex-start', gap: '10px', marginTop: '20px', flexWrap: 'wrap' }}>
            {appState !== 'evaluation_complete' && ( // Only show Submit Answers before evaluation is complete
              <button
                onClick={handleSubmitAnswers}
                disabled={interviewQuestions.some(q => q.answer.trim() === '') || appState === 'evaluating_answers'}
                style={{
                  padding: '10px 20px',
                  backgroundColor: (interviewQuestions.every(q => q.answer.trim() !== '') && appState !== 'evaluating_answers') ? '#0056b3' : '#ccc',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: (interviewQuestions.every(q => q.answer.trim() !== '') && appState !== 'evaluating_answers') ? 'pointer' : 'not-allowed',
                  fontSize: '16px',
                }}
              >
                {appState === 'evaluating_answers' ? 'Evaluating...' : 'Submit Answers'}
              </button>
            )}

            {/* Overall Summary Display */}
            {appState === 'evaluation_complete' && overallSummary && (
                <div style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '20px', width: '100%' }}>
                    <h3 style={{ color: '#333', marginBottom: '10px' }}>Overall Interview Summary:</h3>
                    <div style={{
                        backgroundColor: '#e6f7ff',
                        padding: '15px',
                        borderRadius: '8px',
                        border: '1px solid #a3d9ff',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        maxHeight: '400px',
                        overflowY: 'auto'
                    }}>
                        {overallSummary}
                    </div>
                </div>
            )}
            
            <button
              onClick={handleClearAll}
              style={{
                padding: '10px 20px',
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '16px'
              }}
            >
              Start Over
            </button>
          </div>
        </section>
      )}

      {/* Loading Spinner for Evaluation */}
      {appState === 'evaluating_answers' && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '50px', border: '1px solid #0056b3', borderRadius: '8px',
          flexDirection: 'column', marginBottom: '40px', backgroundColor: '#e0f7fa'
        }}>
          <div className="spinner" style={{
              border: '4px solid rgba(0, 0, 0, 0.1)',
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              borderLeftColor: '#0056b3',
              animation: 'spin 1s ease infinite',
              marginBottom: '15px'
          }}></div>
          <p style={{ fontSize: '1.2em', color: '#0056b3' }}>{currentStatus}</p>
          <style jsx>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}

    </main>
  );
}