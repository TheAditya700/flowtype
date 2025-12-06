import { useState, useCallback, useEffect } from 'react';
import { KeystrokeEvent } from '../types';

interface TypingSessionHook {
  sessionStartTime: number | null;
  sessionDuration: number;
  keystrokeEvents: KeystrokeEvent[];
  startSession: () => void;
  endSession: () => void;
  addKeystrokeEvent: (event: KeystrokeEvent) => void;
}

const useTypingSession = (): TypingSessionHook => {
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionEndTime, setSessionEndTime] = useState<number | null>(null);
  const [keystrokeEvents, setKeystrokeEvents] = useState<KeystrokeEvent[]>([]);
  const [currentTime, setCurrentTime] = useState<number>(Date.now());

  // Live Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (sessionStartTime && !sessionEndTime) {
      interval = setInterval(() => {
        setCurrentTime(Date.now());
      }, 100); // Update every 100ms for smooth WPM
    }
    return () => clearInterval(interval);
  }, [sessionStartTime, sessionEndTime]);

  const startSession = useCallback(() => {
    const now = Date.now();
    setSessionStartTime(now);
    setCurrentTime(now);
    setSessionEndTime(null);
    setKeystrokeEvents([]);
  }, []);

  const endSession = useCallback(() => {
    setSessionEndTime(Date.now());
  }, []);

  const addKeystrokeEvent = useCallback((event: KeystrokeEvent) => {
    setKeystrokeEvents(prevEvents => [...prevEvents, event]);
  }, []);

  const sessionDuration = sessionStartTime && sessionEndTime
    ? (sessionEndTime - sessionStartTime) / 1000
    : sessionStartTime
      ? (currentTime - sessionStartTime) / 1000
      : 0;

  return {
    sessionStartTime,
    sessionDuration,
    keystrokeEvents,
    startSession,
    endSession,
    addKeystrokeEvent,
  };
};

export default useTypingSession;
