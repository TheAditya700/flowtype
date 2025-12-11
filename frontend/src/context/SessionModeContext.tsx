import React, { createContext, useContext, useState } from 'react';

type SessionMode = '15' | '30' | '60' | '120' | 'free';

interface SessionModeContextType {
  sessionMode: SessionMode;
  setSessionMode: (mode: SessionMode) => void;
  sessionStarted: boolean;
  setSessionStarted: (started: boolean) => void;
  isPaused: boolean;
  setIsPaused: (paused: boolean) => void;
}

const SessionModeContext = createContext<SessionModeContextType | undefined>(undefined);

export const SessionModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sessionMode, setSessionMode] = useState<SessionMode>('15');
  const [sessionStarted, setSessionStarted] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(true);

  return (
    <SessionModeContext.Provider value={{ sessionMode, setSessionMode, sessionStarted, setSessionStarted, isPaused, setIsPaused }}>
      {children}
    </SessionModeContext.Provider>
  );
};

export const useSessionMode = (): SessionModeContextType => {
  const context = useContext(SessionModeContext);
  if (!context) {
    throw new Error('useSessionMode must be used within SessionModeProvider');
  }
  return context;
};
