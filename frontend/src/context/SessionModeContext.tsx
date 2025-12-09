import React, { createContext, useContext, useState } from 'react';

type SessionMode = '15' | '30' | '60' | '120' | 'free';

interface SessionModeContextType {
  sessionMode: SessionMode;
  setSessionMode: (mode: SessionMode) => void;
}

const SessionModeContext = createContext<SessionModeContextType | undefined>(undefined);

export const SessionModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sessionMode, setSessionMode] = useState<SessionMode>('free');

  return (
    <SessionModeContext.Provider value={{ sessionMode, setSessionMode }}>
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
