import { useState, useEffect, useRef, useCallback } from 'react';
import TypingZone from './components/TypingZone';
import StatsPanel from './components/StatsPanel';
import { fetchNextSnippet, saveSession } from './api/client';
import { UserState, SnippetLog, KeystrokeEvent, SessionCreateRequest, SnippetResult } from './types';
import { RotateCcw } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid'; // Import uuid for unique ID generation

import PauseScreen from './components/PauseScreen';

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
  const [allKeystrokes, setAllKeystrokes] = useState<KeystrokeEvent[]>([]);
  const [isPaused, setIsPaused] = useState(false);

  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  const [sessionStats, setSessionStats] = useState({
    totalWords: 0,
    totalErrors: 0,
    totalDuration: 0
  });
  
  // Live stats for the current snippet
  const [liveStats, setLiveStats] = useState({ wpm: 0, accuracy: 100 });

  const handleStatsUpdate = useCallback((stats: { wpm: number; accuracy: number }) => {
    setLiveStats({ wpm: stats.wpm, accuracy: stats.accuracy });
  }, []);

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
    setAllKeystrokes(prev => [...prev, ...stats.keystrokeEvents]);
    
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
    const updatedRecentIds = [...recentSnippetIds, currentSnippet.id].slice(-20);
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
    setAllKeystrokes([]);
    setSessionStats({ totalWords: 0, totalErrors: 0, totalDuration: 0 });
    setSessionStartTime(null);
    setIsPaused(false);
    setUserState({
      rollingWpm: 0,
      rollingAccuracy: 1,
      backspaceRate: 0,
      hesitationCount: 0,
      recentErrors: [],
      currentDifficulty: 5,
    });
    // Could also fetch new snippets here to clear queue
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center p-8 relative font-mono">
      
      {/* Pause Screen Overlay */}
      {isPaused && (
        <PauseScreen 
          onResume={() => setIsPaused(false)}
          onReset={resetSession}
          logs={snippetLogs}
          allKeystrokes={allKeystrokes}
          totalWords={sessionStats.totalWords}
          totalTime={sessionStats.totalDuration}
        />
      )}

      {/* Header / Minimal Stats */}
      <div className="w-full max-w-6xl flex justify-between items-end mb-12 opacity-50 hover:opacity-100 transition-opacity">
        <h1 className="text-3xl font-bold text-gray-500">flowtype</h1>
        <div className="flex gap-8 text-xl text-gray-400">
          <div>
            <span className="text-sm text-gray-600 mr-2">wpm</span>
            <span className="font-bold text-blue-400">{Math.round(liveStats.wpm)}</span>
          </div>
          <div>
            <span className="text-sm text-gray-600 mr-2">acc</span>
            <span className="font-bold text-green-400">{Math.round(liveStats.accuracy)}%</span>
          </div>
        </div>
      </div>
      
      {/* Main Typing Area */}
      <div className="w-full max-w-[1600px] px-8 relative">
          {snippetQueue.length > 0 ? (
            <TypingZone 
              snippets={snippetQueue} 
              onSnippetComplete={handleSnippetComplete}
              onRequestPause={() => setIsPaused(true)}
              onStatsUpdate={handleStatsUpdate}
            />
          ) : (
            <div className="text-center text-gray-500 animate-pulse">loading...</div>
          )}
      </div>

      {/* Footer Hints */}
      <div className="absolute bottom-8 text-center text-gray-600 text-sm">
        <span className="bg-gray-800 px-2 py-1 rounded text-xs mr-2">enter</span> to pause
      </div>
    </div>
  );
}

export default App;
