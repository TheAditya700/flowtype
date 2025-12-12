import { useState, useCallback, useEffect, useRef } from 'react';
import { KeystrokeEvent } from '../types';

interface TypingSessionHook {
  sessionStartTime: number | null;
  sessionDuration: number;
  keystrokeEvents: KeystrokeEvent[];
  startSession: () => void;
  endSession: () => void;
  addKeystrokeEvent: (event: KeystrokeEvent) => void;
  updateKeystrokeEvent: (id: string, updates: Partial<KeystrokeEvent>) => void;
}

const useTypingSession = (): TypingSessionHook => {
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionEndTime, setSessionEndTime] = useState<number | null>(null);
  const [keystrokeEvents, setKeystrokeEvents] = useState<KeystrokeEvent[]>([]);

  // Live Duration Tracking
  const [liveDuration, setLiveDuration] = useState(0);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (sessionStartTime && !sessionEndTime) {
      interval = setInterval(() => {
        const now = Date.now();
        const totalElapsed = now - sessionStartTime;
        setLiveDuration(Math.max(0, totalElapsed / 1000));
      }, 100);
    }
    return () => clearInterval(interval);
  }, [sessionStartTime, sessionEndTime]);

  const startSession = useCallback(() => {
    // Only set start time once; keep existing keystrokes so session spans multiple snippets
    if (sessionStartTime) return;
    const now = Date.now();
    setSessionStartTime(now);
    setSessionEndTime(null);
    setLiveDuration(0);
  }, [sessionStartTime]);

  const endSession = useCallback(() => {
    const now = Date.now();
    setSessionEndTime(now);
  }, []);

  const addKeystrokeEvent = useCallback((event: KeystrokeEvent) => {
    setKeystrokeEvents(prevEvents => [...prevEvents, event]);
  }, []);

  const updateKeystrokeEvent = useCallback((id: string, updates: Partial<KeystrokeEvent>) => {
    setKeystrokeEvents(prevEvents => 
      prevEvents.map(ev => (ev.id === id ? { ...ev, ...updates } : ev))
    );
  }, []);

  const sessionDuration = sessionStartTime && sessionEndTime
    ? (sessionEndTime - sessionStartTime) / 1000
    : liveDuration;

  return {
    sessionStartTime,
    sessionDuration,
    keystrokeEvents,
    startSession,
    endSession,
    addKeystrokeEvent,
    updateKeystrokeEvent,
  };
};

export default useTypingSession;
