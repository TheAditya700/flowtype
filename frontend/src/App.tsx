import React, { useState, useEffect, useRef } from 'react';
import TypingZone from './components/TypingZone';
import StatsPanel from './components/StatsPanel';
import { TypingSession, UserState } from './types';
import { fetchNextSnippet, saveSession } from './api/client';

interface QueuedSnippet {
  id: string;
  words: string;
  difficulty: number;
}

function App() {
  const [snippetQueue, setSnippetQueue] = useState<QueuedSnippet[]>([]);
  const [recentSnippetIds, setRecentSnippetIds] = useState<string[]>([]);
  const [userState, setUserState] = useState<UserState>({
    rollingWpm: 0,
    rollingAccuracy: 1,
    backspaceRate: 0,
    hesitationCount: 0,
    recentErrors: [],
    currentDifficulty: 5,
  });
  const [sessionHistory, setSessionHistory] = useState<TypingSession[]>([]);
  const isFetching = useRef(false);

  const fetchMoreSnippets = async (state: UserState, count: number = 1) => {
    if (isFetching.current) return;
    isFetching.current = true;
    try {
      // Track snippets fetched in this batch to avoid duplicates
      const fetchedInBatch: string[] = [];
      
      for (let i = 0; i < count; i++) {
        // Include both recent IDs and IDs fetched in this batch
        const excludeIds = [...recentSnippetIds, ...fetchedInBatch];
        const stateWithRecent = { ...state, recentSnippetIds: excludeIds };
        
        // Pass current snippet ID to ensure next snippet is different
        const currentId = snippetQueue[0]?.id;
        const snippet = await fetchNextSnippet(stateWithRecent, currentId);
        if (snippet) {
          setSnippetQueue(prev => [...prev, snippet]);
          fetchedInBatch.push(snippet.id);
        }
      }
    } catch (error) {
      console.error("Error fetching snippets:", error);
    } finally {
      isFetching.current = false;
    }
  };

  useEffect(() => {
    // Initial fetch: load 2 snippets
    fetchMoreSnippets(userState, 2);
  }, []);

  useEffect(() => {
    // Keep at least 2 snippets in queue (current + next)
    if (snippetQueue.length <= 1) {
      fetchMoreSnippets(userState, 1);
    }
  }, [snippetQueue.length]);

  const handleSnippetComplete = async (session: TypingSession) => {
    console.log("Snippet completed:", session);
    setSessionHistory(prev => [...prev, session]);

    // Add completed snippet ID to recent list (keep last 10)
    setRecentSnippetIds(prev => {
      const updated = [...prev];
      if (snippetQueue[0]?.id) {
        updated.push(snippetQueue[0].id);
      }
      return updated.slice(-10); // Keep only last 10 snippet IDs
    });

    const newUserState = {
      ...userState,
      rollingWpm: session.wpm,
      rollingAccuracy: session.accuracy,
      currentDifficulty: Math.max(1, Math.min(10, userState.currentDifficulty + (session.accuracy > 0.9 ? 0.5 : -0.5))),
    };
    setUserState(newUserState);

    try {
      await saveSession(session);
      console.log("Snippet session saved.");
    } catch (error) {
      console.error("Error saving session:", error);
    }

    // Remove the completed snippet from the queue
    // This makes snippets[1] become the new snippets[0]
    setSnippetQueue(prev => prev.slice(1));

    // Fetch 1 new snippet to replace the one we just removed
    // This ensures we always have at least 2 in queue (current + next)
    fetchMoreSnippets(newUserState, 1);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-5xl font-bold mb-8 text-blue-400">FlowType</h1>
      <div className="flex flex-col md:flex-row gap-8 w-full max-w-4xl">
        <div className="flex-1 bg-gray-800 p-6 rounded-lg shadow-lg">
          {snippetQueue.length > 0 ? (
            <TypingZone snippets={snippetQueue} onSnippetComplete={handleSnippetComplete} />
          ) : (
            <p className="text-center">Loading snippets...</p>
          )}
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
