import React from 'react';

interface WordDisplayProps {
  words: string[];
  currentWordIndex: number;
  charIndex: number;
  inputValue: string;
}

const WordDisplay: React.FC<WordDisplayProps> = ({ words, currentWordIndex, charIndex, inputValue }) => {
  const wordsArray = Array.isArray(words) ? words : [];
  
  return (
    <div className="word-display text-3xl leading-relaxed font-mono">
      {wordsArray.map((word, wordIdx) => (
        <span key={wordIdx} className={`mr-2 ${wordIdx < currentWordIndex ? 'text-gray-500' : 'text-gray-300'}`}>
          {word.split('').map((char, charIdxInWord) => {
            let charClass = '';
            if (wordIdx === currentWordIndex) {
              if (charIdxInWord < charIndex) {
                charClass = char === inputValue[charIdxInWord] ? 'text-green-400' : 'text-red-500';
              } else if (charIdxInWord === charIndex) {
                charClass = 'underline';
              }
            }
            return (
              <span key={charIdxInWord} className={charClass}>
                {char}
              </span>
            );
          })}
        </span>
      ))}
    </div>
  );
};

export default WordDisplay;
