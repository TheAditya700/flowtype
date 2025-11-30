import { useState, useCallback } from 'react';
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

  const startSession = useCallback(() => {
    setSessionStartTime(Date.now());
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
      ? (Date.now() - sessionStartTime) / 1000
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
