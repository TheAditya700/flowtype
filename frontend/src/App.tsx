import React, { useState, useEffect } from 'react';
import TypingZone from './components/TypingZone';
import StatsPanel from './components/StatsPanel';
import { TypingSession, UserState } from './types';
import { fetchNextSnippet, saveSession } from './api/client';

function App() {
  const [currentSnippet, setCurrentSnippet] = useState<string[]>([]);
  const [userState, setUserState] = useState<UserState>({
    rollingWpm: 0,
    rollingAccuracy: 1,
    backspaceRate: 0,
    hesitationCount: 0,
    recentErrors: [],
    currentDifficulty: 5, // Starting difficulty
  });
  const [sessionHistory, setSessionHistory] = useState<TypingSession[]>([]);

  useEffect(() => {
    // Fetch initial snippet
    const getInitialSnippet = async () => {
      try {
        const snippets = await fetchNextSnippet(userState);
        if (snippets && snippets.length > 0) {
          setCurrentSnippet(snippets[0].words.split(' '));
        } else {
          setCurrentSnippet(["no", "snippets", "found"]); // Fallback
        }
      } catch (error) {
        console.error("Error fetching initial snippet:", error);
        setCurrentSnippet(["error", "loading", "snippet"]); // Fallback
      }
    };
    getInitialSnippet();
  }, []); // Run once on mount

  const handleSessionComplete = async (session: TypingSession) => {
    console.log("Session completed:", session);
    setSessionHistory(prev => [...prev, session]);

    // Update user state based on completed session
    const newUserState = {
      ...userState,
      rollingWpm: session.wpm,
      rollingAccuracy: session.accuracy,
      // More sophisticated updates would go here
      currentDifficulty: session.difficultyLevel,
    };
    setUserState(newUserState);

    // Save session to backend
    try {
      await saveSession(session);
      console.log("Session saved to backend.");
    } catch (error) {
      console.error("Error saving session:", error);
    }

    // Fetch next snippet
    try {
      const nextSnippets = await fetchNextSnippet(newUserState); // Use updated userState
      if (nextSnippets && nextSnippets.length > 0) {
        setCurrentSnippet(nextSnippets[0].words.split(' '));
      } else {
        setCurrentSnippet(["no", "more", "snippets"]); // Fallback
      }
    } catch (error) {
      console.error("Error fetching next snippet:", error);
      setCurrentSnippet(["error", "loading", "next", "snippet"]); // Fallback
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-5xl font-bold mb-8 text-blue-400">FlowType</h1>
      <div className="flex flex-col md:flex-row gap-8 w-full max-w-4xl">
        <div className="flex-1 bg-gray-800 p-6 rounded-lg shadow-lg">
          <TypingZone words={currentSnippet} onSessionComplete={handleSessionComplete} />
        </div>
        <div className="w-full md:w-1/3 bg-gray-800 p-6 rounded-lg shadow-lg">
          <StatsPanel wpm={userState.rollingWpm} accuracy={userState.rollingAccuracy * 100} />
          {/* <SessionHistory sessions={sessionHistory} /> */}
        </div>
      </div>
    </div>
  );
}

export default App;
