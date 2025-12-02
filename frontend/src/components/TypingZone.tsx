import React, { useState, useEffect, useRef } from 'react';
import WordDisplay from './WordDisplay';
import useTypingSession from '../hooks/useTypingSession';
import useWPMCalculation from '../hooks/useWPMCalculation';
import { KeystrokeEvent } from '../types';

interface SnippetItem {
  id: string;
  words: string[];
  difficulty: number;
}

interface SnippetStats {
    wpm: number;
    accuracy: number;
    errors: number;
    duration: number;
    keystrokeEvents: KeystrokeEvent[];
    startedAt: Date;
}

interface TypingZoneProps {
  snippets: SnippetItem[];
  onSnippetComplete: (stats: SnippetStats) => void;
}

const TypingZone: React.FC<TypingZoneProps> = ({ snippets, onSnippetComplete }) => {
  const [wordIndex, setWordIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [typed, setTyped] = useState('');
  const [errors, setErrors] = useState(0);
  const [sessionStarted, setSessionStarted] = useState(false);
  const containerRef = useRef<HTMLInputElement>(null);

  // Use local typing session tracking just for this snippet
  const {
    sessionStartTime,
    sessionDuration,
    keystrokeEvents,
    startSession,
    endSession,
    addKeystrokeEvent,
  } = useTypingSession();

  const { wpm, accuracy } = useWPMCalculation(keystrokeEvents, sessionDuration);

  // Always use the first snippet (current) and second snippet (next preview)
  const currentSnippet = snippets[0];
  const nextSnippet = snippets[1];
  const words = currentSnippet?.words || [];

  // Key handler for typing
  useEffect(() => {
    const el = containerRef.current;
    if (!el || !currentSnippet) return;
    el.focus();

    const onKey = (e: KeyboardEvent) => {
      if (!currentSnippet) return;

      if (!sessionStarted) {
        startSession();
        setSessionStarted(true);
      }

      // Backspace
      if (e.key === 'Backspace') {
        e.preventDefault();
        if (typed.length > 0) {
          const newTyped = typed.slice(0, -1);
          setTyped(newTyped);
          setCharIndex(Math.max(0, charIndex - 1));
          addKeystrokeEvent({ timestamp: Date.now(), key: 'Backspace', isBackspace: true, isCorrect: false });
        }
        return;
      }

      // Space -> submit current word
      if (e.key === ' ') {
        e.preventDefault();
        const currentWord = words[wordIndex] || '';
        const typedTrim = typed.trim();
        const correct = typedTrim === currentWord;

        if (!correct) {
          setErrors(prev => prev + Math.max(1, Math.abs(typedTrim.length - currentWord.length)));
        }

        addKeystrokeEvent({ timestamp: Date.now(), key: ' ', isBackspace: false, isCorrect: correct });

        // If last word of snippet, complete snippet
        if (wordIndex === words.length - 1) {
          const finalDuration = sessionDuration; // Capture current duration
          endSession(); // Mark end
          
          const stats: SnippetStats = {
            wpm,
            accuracy: accuracy / 100,
            errors,
            duration: finalDuration,
            keystrokeEvents,
            startedAt: new Date(sessionStartTime || Date.now())
          };
          
          onSnippetComplete(stats);

          // Reset for next snippet (parent will remove current and shift queue)
          // The new currentSnippet (from snippets[1]) will become snippets[0]
          setWordIndex(0);
          setTyped('');
          setCharIndex(0);
          setErrors(0);
          setSessionStarted(false);
          return;
        }

        // advance to next word within snippet
        setWordIndex(i => i + 1);
        setTyped('');
        setCharIndex(0);
        return;
      }

      // Ignore non-printable keys
      if (e.key.length !== 1) return;

      // Normal character
      e.preventDefault();
      const newTyped = typed + e.key;
      setTyped(newTyped);
      setCharIndex(i => i + 1);

      const expectedChar = (words[wordIndex] || '')[charIndex] || '';
      const isCorrect = e.key === expectedChar;
      addKeystrokeEvent({ timestamp: Date.now(), key: e.key, isBackspace: false, isCorrect });
    };

    el.addEventListener('keydown', onKey as any);
    return () => el.removeEventListener('keydown', onKey as any);
  }, [typed, charIndex, wordIndex, words, sessionStarted, currentSnippet, startSession, endSession, addKeystrokeEvent, keystrokeEvents, sessionDuration, wpm, accuracy, errors, onSnippetComplete]);

  if (!currentSnippet) {
    return <p className="text-center text-gray-400">Loading...</p>;
  }

  return (
    <div className="typing-zone relative">
      {/* Display current snippet and next preview */}
      <div className="space-y-4 overflow-hidden">
        {currentSnippet && (
          <div className="p-4 rounded-lg bg-gray-700 border-2 border-blue-500">
            <WordDisplay
              words={words}
              currentWordIndex={wordIndex}
              charIndex={charIndex}
              inputValue={typed}
            />
          </div>
        )}
        {nextSnippet && (
          <div className="p-4 rounded-lg bg-gray-800 opacity-60 border border-gray-600">
            <WordDisplay
              words={nextSnippet.words || []}
              currentWordIndex={-1}
              charIndex={0}
              inputValue=""
            />
          </div>
        )}
      </div>

      {/* Input field that captures all keyboard events */}
      <input
        ref={containerRef}
        type="text"
        value={typed}
        onChange={() => {}} // Controlled by keydown, not onChange
        className="w-full mt-4 p-3 bg-gray-900 text-gray-400 rounded-md cursor-text focus:outline-none focus:ring-2 focus:ring-blue-500"
        placeholder="Start typing..."
        autoFocus
        autoCapitalize="off"
        autoCorrect="off"
        spellCheck="false"
      />

      {/* Stats line */}
      <div className="mt-4 text-sm text-gray-400 space-y-1">
        <p>WPM: {wpm.toFixed(0)} | Accuracy: {accuracy.toFixed(0)}% | Errors: {errors}</p>
        <p>Queue: {snippets.length} snippets</p>
      </div>
    </div>
  );
};

export default TypingZone;
