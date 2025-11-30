import React, { useState, useEffect, useRef } from 'react';
import WordDisplay from './WordDisplay';
import useTypingSession from '../hooks/useTypingSession';
import useKeystrokeTracking from '../hooks/useKeystrokeTracking';
import useWPMCalculation from '../hooks/useWPMCalculation';
import { KeystrokeEvent, TypingSession } from '../types';

interface TypingZoneProps {
  words: string[];
  onSessionComplete: (session: TypingSession) => void;
}

const TypingZone: React.FC<TypingZoneProps> = ({ words, onSessionComplete }) => {
  const [inputValue, setInputValue] = useState<string>('');
  const [currentWordIndex, setCurrentWordIndex] = useState<number>(0);
  const [charIndex, setCharIndex] = useState<number>(0);
  const [errors, setErrors] = useState<number>(0);
  const [sessionStarted, setSessionStarted] = useState<boolean>(false);
  const [sessionEnded, setSessionEnded] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const {
    sessionStartTime,
    sessionDuration,
    keystrokeEvents,
    startSession,
    endSession,
    addKeystrokeEvent,
  } = useTypingSession();

  const { wpm, accuracy } = useWPMCalculation(keystrokeEvents, sessionDuration);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [sessionStarted]);

  useEffect(() => {
    if (sessionEnded) {
      const completedSession: TypingSession = {
        startedAt: new Date(sessionStartTime!),
        durationSeconds: sessionDuration,
        wordsTyped: words.length,
        errors: errors,
        wpm: wpm,
        accuracy: accuracy / 100, // Convert to 0-1 range
        difficultyLevel: 5, // Placeholder, will be dynamic
        keystrokeData: keystrokeEvents,
      };
      onSessionComplete(completedSession);
      resetSession();
    }
  }, [sessionEnded]);

  const resetSession = () => {
    setInputValue('');
    setCurrentWordIndex(0);
    setCharIndex(0);
    setErrors(0);
    setSessionStarted(false);
    setSessionEnded(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (words.length === 0) {
      return;
    }

    if (!sessionStarted && words.length > 0) {
      startSession();
      setSessionStarted(true);
    }

    const typedChar = event.key;
    const currentWord = words[currentWordIndex];
    const expectedChar = currentWord ? currentWord[charIndex] : '';

    let isCorrect = false;
    let isBackspace = false;

    if (typedChar === 'Backspace') {
      isBackspace = true;
      if (charIndex > 0) {
        setCharIndex(prev => prev - 1);
      } else if (currentWordIndex > 0) {
        setCurrentWordIndex(prev => prev - 1);
        setInputValue(words[currentWordIndex - 1] + ' ');
        setCharIndex(words[currentWordIndex - 1].length);
      }
    } else if (typedChar.length === 1) { // Only process single character inputs
      if (typedChar === expectedChar) {
        isCorrect = true;
        setCharIndex(prev => prev + 1);
      } else {
        setErrors(prev => prev + 1);
      }

      if (charIndex + 1 > currentWord.length) { // Word completed
        if (typedChar === ' ') {
          setCurrentWordIndex(prev => prev + 1);
          setCharIndex(0);
          setInputValue('');
        } else {
          // Handle extra characters typed after word completion
          setErrors(prev => prev + 1);
        }
      }
    }

    addKeystrokeEvent({
      timestamp: Date.now(),
      key: typedChar,
      isBackspace: isBackspace,
      isCorrect: isCorrect,
    });

    // Check if session is complete
    if (currentWordIndex === words.length - 1 && charIndex === currentWord.length && typedChar === ' ') {
      endSession();
      setSessionEnded(true);
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  return (
    <div className="typing-zone p-4 bg-gray-700 rounded-lg">
      <WordDisplay
        words={words}
        currentWordIndex={currentWordIndex}
        charIndex={charIndex}
        inputValue={inputValue}
      />
      <input
        ref={inputRef}
        type="text"
        className="w-full p-3 mt-4 text-xl bg-gray-900 text-blue-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        value={inputValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={sessionEnded}
        autoFocus
        autoCapitalize="off"
        autoCorrect="off"
        spellCheck="false"
      />
      {sessionEnded && (
        <button
          onClick={resetSession}
          className="mt-4 px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-lg"
        >
          Start New Session
        </button>
      )}
    </div>
  );
};

export default TypingZone;
