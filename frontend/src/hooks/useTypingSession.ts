import { useState, useCallback, useEffect, useRef } from 'react';
import { KeystrokeEvent } from '../types';

interface TypingSessionHook {
  sessionStartTime: number | null;
  sessionDuration: number;
  keystrokeEvents: KeystrokeEvent[];
  startSession: () => void;
  endSession: () => void;
  pauseSession: () => void;
  resumeSession: () => void;
  addKeystrokeEvent: (event: KeystrokeEvent) => void;
  updateKeystrokeEvent: (id: string, updates: Partial<KeystrokeEvent>) => void;
  isPaused: boolean;
}

const useTypingSession = (): TypingSessionHook => {
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionEndTime, setSessionEndTime] = useState<number | null>(null);
  const [keystrokeEvents, setKeystrokeEvents] = useState<KeystrokeEvent[]>([]);
  
  // Pause State
  const [isPaused, setIsPaused] = useState(false);
  const [pauseTime, setPauseTime] = useState<number | null>(null);
  const [totalPausedDuration, setTotalPausedDuration] = useState(0);

  // Live Duration Tracking
  const [liveDuration, setLiveDuration] = useState(0);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (sessionStartTime && !sessionEndTime && !isPaused) {
      interval = setInterval(() => {
        const now = Date.now();
        const totalElapsed = now - sessionStartTime;
        const activeDuration = (totalElapsed - totalPausedDuration) / 1000;
        setLiveDuration(Math.max(0, activeDuration));
      }, 100);
    }
    return () => clearInterval(interval);
  }, [sessionStartTime, sessionEndTime, isPaused, totalPausedDuration]);

  const startSession = useCallback(() => {
    const now = Date.now();
    setSessionStartTime(now);
    setSessionEndTime(null);
    setKeystrokeEvents([]);
    setIsPaused(false);
    setPauseTime(null);
    setTotalPausedDuration(0);
    setLiveDuration(0);
  }, []);

  const pauseSession = useCallback(() => {
    if (!isPaused && sessionStartTime && !sessionEndTime) {
      setIsPaused(true);
      setPauseTime(Date.now());
    }
  }, [isPaused, sessionStartTime, sessionEndTime]);

  const resumeSession = useCallback(() => {
    if (isPaused && pauseTime) {
      const now = Date.now();
      const pausedDuration = now - pauseTime;
      setTotalPausedDuration(prev => prev + pausedDuration);
      setIsPaused(false);
      setPauseTime(null);
    }
  }, [isPaused, pauseTime]);

  const endSession = useCallback(() => {
    const now = Date.now();
    setSessionEndTime(now);
    setIsPaused(false);
    
    // Final calculation if ending while paused? 
    // Usually we end when finishing typing, so not paused.
    // But if we were paused, we should add that last chunk.
    if (isPaused && pauseTime) {
        const pausedDuration = now - pauseTime;
        setTotalPausedDuration(prev => prev + pausedDuration);
    }
  }, [isPaused, pauseTime]);

  const addKeystrokeEvent = useCallback((event: KeystrokeEvent) => {
    setKeystrokeEvents(prevEvents => [...prevEvents, event]);
  }, []);

  const updateKeystrokeEvent = useCallback((id: string, updates: Partial<KeystrokeEvent>) => {
    setKeystrokeEvents(prevEvents => 
      prevEvents.map(ev => (ev.id === id ? { ...ev, ...updates } : ev))
    );
  }, []);

  const sessionDuration = sessionStartTime && sessionEndTime
    ? (sessionEndTime - sessionStartTime - totalPausedDuration) / 1000
    : liveDuration;

  return {
    sessionStartTime,
    sessionDuration,
    keystrokeEvents,
    startSession,
    endSession,
    pauseSession,
    resumeSession,
    addKeystrokeEvent,
    updateKeystrokeEvent,
    isPaused
  };
};

export default useTypingSession;
