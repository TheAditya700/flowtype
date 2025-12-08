import React, { useState, useEffect, useRef, useLayoutEffect } from 'react';
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
  onRequestPause: () => void;
  onStatsUpdate?: (stats: { wpm: number; accuracy: number; time: number }) => void;
}

const TypingZone: React.FC<TypingZoneProps> = ({ snippets, onSnippetComplete, onRequestPause, onStatsUpdate }) => {
  const [typedHistory, setTypedHistory] = useState<string[]>([]);
  const [currentTyped, setCurrentTyped] = useState(''); 
  const [errors, setErrors] = useState(0);
  const [sessionStarted, setSessionStarted] = useState(false);
  
  // Derived State
  const wordIndex = typedHistory.length;
  const charIndex = currentTyped.length;

  // Cursor State
  const [cursorPos, setCursorPos] = useState({ top: 0, left: 0 });
  
  // Input Ref (Hidden)
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Current Data
  const currentSnippet = snippets[0];
  const words = currentSnippet?.words || [];

  // Session Hooks
  const {
    sessionStartTime,
    sessionDuration,
    keystrokeEvents,
    startSession,
    endSession,
    pauseSession, // New
    addKeystrokeEvent,
    updateKeystrokeEvent
  } = useTypingSession();

  const { wpm, accuracy } = useWPMCalculation(keystrokeEvents, sessionDuration);
  
  // Key Tracking for Rollover
  const pendingKeys = useRef<Record<string, string>>({});

  // If there's no snippet, don't execute any further logic or render
  if (!currentSnippet) return null;

  // DEBUGGING: Expose state to window
  useEffect(() => {
    (window as any).__FLOWTYPE_DEBUG__ = {
      wordIndex,
      charIndex,
      currentTyped,
      typedHistory,
      cursorPos,
      snippet: currentSnippet
    };
  }, [wordIndex, charIndex, currentTyped, typedHistory, cursorPos, currentSnippet]);

  // Cursor Positioning Logic (Same as before, but depends on new state)
  useLayoutEffect(() => {
    const updateCursorPosition = () => {
      // Get the current word element
      const wordElements = containerRef.current?.querySelectorAll('.word-container');
      if (!wordElements || wordIndex >= wordElements.length) return;
      
      const currentWordEl = wordElements[wordIndex] as HTMLElement;
      if (!currentWordEl) return;
      
      // Get all character spans in the current word
      const charSpans = currentWordEl.querySelectorAll('.char-span');
      
      // Position cursor at the current character index
      if (charIndex < charSpans.length) {
        // Cursor before this character
        const targetChar = charSpans[charIndex] as HTMLElement;
        const rect = targetChar.getBoundingClientRect();
        const containerRect = containerRef.current!.getBoundingClientRect();
        
        setCursorPos({
          top: rect.top - containerRect.top,
          left: rect.left - containerRect.left
        });
      } else if (charSpans.length > 0) {
        // Cursor after the last character (end of word)
        const lastChar = charSpans[charSpans.length - 1] as HTMLElement;
        const rect = lastChar.getBoundingClientRect();
        const containerRect = containerRef.current!.getBoundingClientRect();
        
        setCursorPos({
          top: rect.top - containerRect.top,
          left: rect.left - containerRect.left + rect.width
        });
      } else {
          // Empty word (start of word), use word container position
          const rect = currentWordEl.getBoundingClientRect();
          const containerRect = containerRef.current!.getBoundingClientRect();
          setCursorPos({
             top: rect.top - containerRect.top,
             left: rect.left - containerRect.left
          });
      }
    };
    
    updateCursorPosition();
    const timeoutId = setTimeout(updateCursorPosition, 0);
    return () => clearTimeout(timeoutId);
  }, [wordIndex, charIndex, currentTyped, currentSnippet]); 

  // Call onStatsUpdate whenever wpm or accuracy changes
  useEffect(() => {
    if (onStatsUpdate) {
      onStatsUpdate({ wpm, accuracy, time: sessionDuration });
    }
  }, [wpm, accuracy, sessionDuration, onStatsUpdate]);

  // Focus Handler
  const handleFocus = () => {
    inputRef.current?.focus();
  };

  // Autofocus
  useEffect(() => {
    inputRef.current?.focus();
  }, [snippets]);

  const handleKeyUp = (e: React.KeyboardEvent<HTMLInputElement>) => {
      const id = pendingKeys.current[e.code];
      if (id) {
          updateKeystrokeEvent(id, { keyup_timestamp: Date.now() });
          delete pendingKeys.current[e.code];
      }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Pause
    if (e.key === 'Enter') {
      e.preventDefault();
      pauseSession(); // Pause tracking immediately
      onRequestPause(); // Notify parent to switch view
      return;
    }

    // Start Session - Only on valid characters
    if (!sessionStarted) {
       // Allow letters, numbers, punctuation, symbols (basically length 1 strings)
       // Do NOT start on Backspace, Shift, Control, Alt, etc.
       const isValidStartKey = e.key.length === 1; 
       
       if (isValidStartKey) {
          startSession();
          setSessionStarted(true);
       } else {
          // If trying to backspace before starting or hitting modifier, just ignore or handle normally
          // but don't start timer.
       }
    }
    
    // Generate Event ID
    const eventId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    pendingKeys.current[e.code] = eventId;

    const currentTargetWord = words[wordIndex] || '';

    // Backspace
    if (e.key === 'Backspace') {
      if (currentTyped.length > 0) {
        // Simple backspace within current word
        setCurrentTyped(prev => prev.slice(0, -1));
        if (sessionStarted) { // Only record if session started
            addKeystrokeEvent({ id: eventId, timestamp: Date.now(), key: 'Backspace', isBackspace: true, isCorrect: false });
        }
      } else if (typedHistory.length > 0) {
        // Backspace to previous word
        const prevWordTyped = typedHistory[typedHistory.length - 1];
        setTypedHistory(prev => prev.slice(0, -1));
        setCurrentTyped(prevWordTyped); // Load previous word back into editor
        // We don't necessarily record a backspace event for "word navigation" but we can if we want strict accounting.
        // MonkeyType usually just lets you edit.
      }
      return;
    }

    // Space (Submit Word)
    if (e.key === ' ') {
      e.preventDefault();
      
      // Logic: Compare typed vs currentWord
      const isCorrectWord = currentTyped === currentTargetWord;
      if (!isCorrectWord) {
        setErrors(prev => prev + 1); 
      }
      
      // Push to history
      const newHistory = [...typedHistory, currentTyped];
      setTypedHistory(newHistory);
      setCurrentTyped('');
      
      if (sessionStarted) {
         addKeystrokeEvent({ id: eventId, timestamp: Date.now(), key: ' ', isBackspace: false, isCorrect: isCorrectWord });
      }

      // Check Completion
      if (newHistory.length === words.length) {
        // Snippet Complete
        const finalDuration = sessionDuration;
        endSession();
        
        onSnippetComplete({
          wpm,
          accuracy: accuracy / 100,
          errors,
          duration: finalDuration,
          keystrokeEvents,
          startedAt: new Date(sessionStartTime || Date.now())
        });
        
        // Reset
        setTypedHistory([]);
        setCurrentTyped('');
        setErrors(0);
        setSessionStarted(false);
      }
      return;
    }

    // Normal Character
    if (e.key.length === 1) {
      const expectedChar = currentTargetWord[charIndex];
      const newTyped = currentTyped + e.key;
      
      // Strict Mode: Don't allow typing more than word length + extra buffer? 
      // Or allow it and show red? User asked for "entire word red" if early space.
      // Let's allow typing freely.
      
      setCurrentTyped(newTyped);
      
      const isCorrect = e.key === expectedChar;
      if (sessionStarted || !sessionStarted) { 
          // Note: If we just started the session above, sessionStarted state might not be updated yet in this render cycle
          // But we want to record the first keystroke too.
          // Since we check isValidStartKey above, we know this is valid.
          addKeystrokeEvent({ id: eventId, timestamp: Date.now(), key: e.key, isBackspace: false, isCorrect });
      }
    }
  };

  // Render Helper
  const renderSnippet = (snippetWords: string[], isCurrentSnippet: boolean) => {
    if (!snippetWords) return null; // Guard against undefined snippetWords

    return (
      <div className={`flex flex-wrap content-start select-none mb-4 ${isCurrentSnippet ? '' : 'opacity-40 grayscale blur-[1px]'}`}>
        {snippetWords.map((targetWord, wIdx) => {
          const isCurrentWord = isCurrentSnippet && wIdx === wordIndex;
          const isPastWord = isCurrentSnippet && wIdx < wordIndex;
          
          // Determine what was typed for this word
          let typedContent = '';
          if (isPastWord) {
              typedContent = typedHistory[wIdx] || '';
          } else if (isCurrentWord) {
              typedContent = currentTyped;
          }

          // Split word into characters
          const chars = targetWord.split('');
          
          // Render extra typed chars (if any)
          const displayChars = (isCurrentWord || isPastWord) && typedContent.length > targetWord.length 
            ? [...chars, ...typedContent.slice(targetWord.length).split('')]
            : chars;

          return (
            <div key={wIdx} className="word-container relative inline-block mr-4 mb-2 text-3xl font-mono leading-relaxed">
              {displayChars.map((char, cIdx) => {
                let colorClass = 'text-gray-600'; 
                let isExtra = cIdx >= targetWord.length; // Is this an extra char typed by user?

                if (!isCurrentSnippet) {
                  colorClass = 'text-gray-500'; 
                } else if (isPastWord) {
                   // PAST WORD LOGIC: Show Errors Red, Correct White
                   const typedChar = typedContent[cIdx];
                   const originalChar = targetWord[cIdx]; // undefined if extra
                   
                   if (typedChar === undefined) {
                       // Missing char (Early Space)
                       colorClass = 'text-red-500'; // Or underline?
                   } else if (isExtra) {
                       // Extra char typed
                       colorClass = 'text-red-600 opacity-70'; 
                   } else if (typedChar === originalChar) {
                       colorClass = 'text-gray-100';
                   } else {
                       colorClass = 'text-red-500';
                   }

                } else if (isCurrentWord) {
                   // CURRENT WORD LOGIC
                  if (cIdx < typedContent.length) {
                    const typedChar = typedContent[cIdx];
                    const originalChar = targetWord[cIdx];
                    
                    if (isExtra) {
                        colorClass = 'text-red-600 opacity-70';
                    } else if (typedChar === originalChar) {
                      colorClass = 'text-gray-100'; 
                    } else {
                      colorClass = 'text-red-500';
                    }
                  } else {
                    colorClass = 'text-gray-600';
                  }
                }

                return (
                  <span 
                    key={cIdx}
                    className={`char-span inline-block ${colorClass} transition-colors duration-75`}
                  >
                    {char}
                  </span>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  if (!currentSnippet) return null;

  return (
    <div 
      className="relative outline-none min-h-[300px] cursor-text flex flex-col" 
      onClick={handleFocus}
      ref={containerRef}
    >
      {/* Hidden Input for capturing typing */}
      <input
        ref={inputRef}
        type="text"
        className="absolute opacity-0 top-0 left-0 h-0 w-0"
        autoFocus
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        value="" 
        onChange={() => {}}
      />

      {/* Cursor Element - Show if session started OR if it's the initial state (word 0, char 0) */}
      {(sessionStarted || (wordIndex === 0 && charIndex === 0)) && cursorPos && (
        <div 
          className="absolute w-1 h-8 bg-blue-400 rounded-full transition-all duration-100 ease-out z-10"
          style={{ 
            top: cursorPos.top + 4, 
            left: cursorPos.left - 2 
          }}
        />
      )}

      {/* Current Snippet */}
      {renderSnippet(words, true)}

      {/* Next Snippet Preview */}
      {snippets[1] && snippets[1].words && renderSnippet(snippets[1].words, false)}
    </div>
  );
};

export default TypingZone;
