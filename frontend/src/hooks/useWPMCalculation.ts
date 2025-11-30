import { useState, useEffect } from 'react';
import { KeystrokeEvent } from '../types';

const useWPMCalculation = (keystrokeEvents: KeystrokeEvent[], sessionDuration: number) => {
  const [wpm, setWpm] = useState<number>(0);
  const [accuracy, setAccuracy] = useState<number>(100); // Percentage

  useEffect(() => {
    if (sessionDuration === 0 || keystrokeEvents.length === 0) {
      setWpm(0);
      setAccuracy(100);
      return;
    }

    let correctChars = 0;
    let totalTypedChars = 0; // Includes correct and incorrect, excludes backspaces

    keystrokeEvents.forEach(event => {
      if (!event.isBackspace && event.key.length === 1) { // Only count actual typed characters
        totalTypedChars++;
        if (event.isCorrect) {
          correctChars++;
        }
      }
    });

    // WPM calculation: (characters / 5) / minutes
    const minutes = sessionDuration / 60;
    const calculatedWPM = (totalTypedChars / 5) / minutes;
    setWpm(isNaN(calculatedWPM) ? 0 : calculatedWPM);

    // Accuracy calculation: (correct characters / total typed characters) * 100
    const calculatedAccuracy = totalTypedChars === 0 ? 100 : (correctChars / totalTypedChars) * 100;
    setAccuracy(isNaN(calculatedAccuracy) ? 100 : calculatedAccuracy);

  }, [keystrokeEvents, sessionDuration]);

  return { wpm, accuracy };
};

export default useWPMCalculation;
