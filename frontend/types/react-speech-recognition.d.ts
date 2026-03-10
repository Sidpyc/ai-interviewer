declare module 'react-speech-recognition' {
  interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
    resultIndex: number;
  }

  interface SpeechRecognitionResultList {
    length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }

  interface SpeechRecognitionResult {
    isFinal: boolean;
    length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
  }

  interface SpeechRecognitionAlternative {
    transcript: string;
    confidence: number;
  }

  interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start(): void;
    stop(): void;
    abort(): void;
    onresult: (event: SpeechRecognitionEvent) => void;
    onerror: (event: Event) => void;
    onend: () => void;
    onstart: () => void;
  }

  interface UseSpeechRecognitionResult {
    transcript: string;
    interimTranscript: string;
    finalTranscript: string;
    listening: boolean;
    browserSupportsSpeechRecognition: boolean;
    isMicrophoneAvailable: boolean | 'checking';
    startListening: (options?: { continuous?: boolean; language?: string }) => void;
    stopListening: () => void;
    resetTranscript: () => void;
  }

  const SpeechRecognition: {
    new (): SpeechRecognition;
    prototype: SpeechRecognition;
    startListening: (options?: { continuous?: boolean; language?: string }) => void;
    stopListening: () => void;
  };

  function useSpeechRecognition(): UseSpeechRecognitionResult;

  export default SpeechRecognition;
  export { useSpeechRecognition };
}
