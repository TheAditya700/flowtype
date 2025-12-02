import { useState, useEffect, useRef } from 'react';
import TypingZone from './components/TypingZone';
import StatsPanel from './components/StatsPanel';
import { fetchNextSnippet, saveSession } from './api/client';
import { UserState, SnippetLog, KeystrokeEvent, SessionCreateRequest, SnippetResult } from './types';
import { RotateCcw } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid'; // Import uuid for unique ID generation

interface QueuedSnippet {
  id: string;
  words: string[];
  difficulty: number;
}

function App() {
  const [snippetQueue, setSnippetQueue] = useState<QueuedSnippet[]>([]);
  const [recentSnippetIds, setRecentSnippetIds] = useState<string[]>([]);
  const [userState, setUserState] = useState<UserState>(() => {
    // Initialize user ID from localStorage or generate a new one
    let userId = localStorage.getItem('flowtype_user_id');
    if (!userId) {
      userId = uuidv4();
      localStorage.setItem('flowtype_user_id', userId);
    }
    return {
      user_id: userId, // Set the persistent user ID here
      rollingWpm: 0,
      rollingAccuracy: 1,
      backspaceRate: 0,
      hesitationCount: 0,
      recentErrors: [],
      currentDifficulty: 5,
    };
  });
  const [snippetLogs, setSnippetLogs] = useState<SnippetLog[]>([]);
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  const [sessionStats, setSessionStats] = useState({
    totalWords: 0,
    totalErrors: 0,
    totalDuration: 0
  });

  const isFetching = useRef(false);

  const fetchMoreSnippets = async (state: UserState, count: number = 1, overrideRecentIds?: string[]) => {
    if (isFetching.current) return;
    isFetching.current = true;
    try {
      const fetchedInBatch: string[] = [];
      for (let i = 0; i < count; i++) {
        // Use override if provided, else current state
        const currentRecentIds = overrideRecentIds || recentSnippetIds;
        const excludeIds = [...currentRecentIds, ...fetchedInBatch];
        // Explicitly attach recentSnippetIds to state sent to backend
        const stateWithRecent = { ...state, recentSnippetIds: excludeIds };
        
        const currentId = snippetQueue.length > 0 ? snippetQueue[0]?.id : undefined;
        const snippet = await fetchNextSnippet(stateWithRecent, currentId);
        if (!snippet) {
          console.log("No new snippet found after filtering. Stopping fetch.");
          break; // Stop fetching if no new snippet is found
        }
        setSnippetQueue(prev => [...prev, snippet]);
        fetchedInBatch.push(snippet.id);
      }
    } catch (error) {
      console.error("Error fetching snippets:", error);
    } finally {
      isFetching.current = false;
    }
  };

  useEffect(() => {
    fetchMoreSnippets(userState, 2);
  }, []);

  useEffect(() => {
    if (snippetQueue.length <= 1) {
      fetchMoreSnippets(userState, 1);
    }
  }, [snippetQueue.length]);

  // Initialize session start time on first interaction if needed,
  // but typically we'll set it when the first snippet completes if null,
  // or we can assume session starts when the user lands?
  // Better: Set it when the first snippet log is added if it's null.

  const handleSnippetComplete = (stats: { 
    wpm: number; 
    accuracy: number; 
    errors: number; 
    duration: number; 
    keystrokeEvents: KeystrokeEvent[]; 
    startedAt: Date 
  }) => {
    if (!sessionStartTime) {
        setSessionStartTime(stats.startedAt);
    }

    const currentSnippet = snippetQueue[0];
    if (!currentSnippet) return;

    console.log("Snippet completed:", stats);

    const safeDifficulty = currentSnippet.difficulty ?? 5.0;

    // Create log
    const log: SnippetLog = {
        snippet_id: currentSnippet.id,
        started_at: stats.startedAt.toISOString(),
        completed_at: new Date().toISOString(),
        wpm: stats.wpm,
        accuracy: stats.accuracy,
        difficulty: safeDifficulty
    };

    setSnippetLogs(prev => [...prev, log]);
    setSessionStats(prev => ({
        totalWords: prev.totalWords + currentSnippet.words.length,
        totalErrors: prev.totalErrors + stats.errors,
        totalDuration: prev.totalDuration + stats.duration
    }));

    // Update User State
    const newUserState = {
      ...userState,
      rollingWpm: stats.wpm,
      rollingAccuracy: stats.accuracy,
      currentDifficulty: Math.max(1, Math.min(10, userState.currentDifficulty + (stats.accuracy > 0.9 ? 0.5 : -0.5))),
    };
    setUserState(newUserState);
    
    // Add to recent IDs
    const updatedRecentIds = [...recentSnippetIds, currentSnippet.id].slice(-10);
    setRecentSnippetIds(updatedRecentIds);

    // Shift queue
    setSnippetQueue(prev => prev.slice(1));
    // Fetch with updated exclusion list to prevent immediate repeat
    fetchMoreSnippets(newUserState, 1, updatedRecentIds);

    // Construct full session payload
    // We treat each snippet completion as a "session" for now to get granular updates
    const snippetResult: SnippetResult = {
        snippet_id: currentSnippet.id,
        wpm: stats.wpm,
        accuracy: stats.accuracy,
        difficulty: safeDifficulty,
        started_at: stats.startedAt.getTime(),
        completed_at: Date.now()
    };

    const sessionPayload: SessionCreateRequest = {
        user_id: userState.user_id,
        durationSeconds: stats.duration,
        wordsTyped: currentSnippet.words.length, // or calculate from keystrokes
        keystrokeData: stats.keystrokeEvents,
        wpm: stats.wpm,
        accuracy: stats.accuracy,
        errors: stats.errors,
        difficultyLevel: safeDifficulty,
        snippets: [snippetResult],
        user_state: userState, // Send PREVIOUS state so backend knows context? Or new? Usually previous state + new events -> new state.
        flowScore: 0.0 // Calculated on backend usually?
    };

    // Send to backend
    saveSession(sessionPayload).catch(err => console.error("Failed to save session:", err));
  };

  // Manual session save removed: telemetry is sent per-snippet.
  // Session state can still be reset by the user or by a new session start.

  const resetSession = () => {
    setSnippetLogs([]);
    setSessionStats({ totalWords: 0, totalErrors: 0, totalDuration: 0 });
    setSessionStartTime(null);
    setUserState({
      rollingWpm: 0,
      rollingAccuracy: 1,
      backspaceRate: 0,
      hesitationCount: 0,
      recentErrors: [],
      currentDifficulty: 5,
    });
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
        <div className="w-full md:w-1/3 bg-gray-800 p-6 rounded-lg shadow-lg flex flex-col gap-4">
          <StatsPanel wpm={userState.rollingWpm} accuracy={userState.rollingAccuracy * 100} />
          
          <div className="bg-gray-700 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2 text-gray-300">Session Progress</h3>
            <div className="text-sm text-gray-400 space-y-1">
                <p>Snippets Completed: {snippetLogs.length}</p>
                <p>Total Words: {sessionStats.totalWords}</p>
                <p>Total Time: {Math.round(sessionStats.totalDuration)}s</p>
            </div>
            {snippetLogs.length > 0 && (
              <button 
                onClick={resetSession}
                className="mt-4 w-full flex items-center justify-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white py-2 px-4 rounded transition-colors"
              >
                <RotateCcw size={18} />
                Reset Session
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
