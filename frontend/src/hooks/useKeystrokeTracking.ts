import { useEffect, useCallback } from 'react';
import { KeystrokeEvent } from '../types';

interface KeystrokeTrackingHook {
  // No explicit return values needed for this hook, it primarily handles side effects
}

const useKeystrokeTracking = (
  onKeystroke: (event: KeystrokeEvent) => void,
  enabled: boolean = true
): KeystrokeTrackingHook => {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled) return;

    const keystroke: KeystrokeEvent = {
      timestamp: Date.now(),
      key: event.key,
      isBackspace: event.key === 'Backspace',
      isCorrect: false, // This will be determined by the TypingZone logic
    };
    onKeystroke(keystroke);
  }, [onKeystroke, enabled]);

  useEffect(() => {
    if (enabled) {
      window.addEventListener('keydown', handleKeyDown);
    } else {
      window.removeEventListener('keydown', handleKeyDown);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown, enabled]);

  return {};
};

export default useKeystrokeTracking;
