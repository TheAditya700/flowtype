import { useState, useEffect, useRef } from 'react';
import { KeystrokeEvent } from '../types';

const useWPMCalculation = (keystrokeEvents: KeystrokeEvent[], sessionDuration: number) => {
  const latestWpm = useRef(0);
  const latestAccuracy = useRef(100);

  const [displayedWpm, setDisplayedWpm] = useState<number>(0);
  const [displayedAccuracy, setDisplayedAccuracy] = useState<number>(100);

  // Effect to calculate WPM/Accuracy frequently (this runs on every keystroke/timer update)
  useEffect(() => {
    if (sessionDuration === 0 || keystrokeEvents.length === 0) {
      latestWpm.current = 0;
      latestAccuracy.current = 100;
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
    let calculatedWPM = (totalTypedChars / 5) / minutes;
    if (isNaN(calculatedWPM) || !isFinite(calculatedWPM)) calculatedWPM = 0;

    // Accuracy calculation: (correct characters / total typed characters) * 100
    const calculatedAccuracy = totalTypedChars === 0 ? 100 : (correctChars / totalTypedChars) * 100;
    
    latestWpm.current = calculatedWPM;
    latestAccuracy.current = calculatedAccuracy;

  }, [keystrokeEvents, sessionDuration]);

  // Effect to update displayed WPM/Accuracy every second
  useEffect(() => {
    const interval = setInterval(() => {
      setDisplayedWpm(Math.round(latestWpm.current));
      setDisplayedAccuracy(Math.round(latestAccuracy.current));
    }, 1000); // Update every 1 second

    // Clear interval on cleanup or if session ends
    return () => clearInterval(interval);
  }, []); // Only run once on mount

  return { wpm: displayedWpm, accuracy: displayedAccuracy };
};

export default useWPMCalculation;
