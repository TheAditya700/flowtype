import { useState, useEffect, useRef, useCallback } from 'react';
import TypingZone from './components/TypingZone';
import { fetchNextSnippet, saveSession } from './api/client';
import { UserState, SnippetLog, KeystrokeEvent, SessionCreateRequest, SnippetResult, SessionResponse } from './types'; // Import SessionResponse
import { v4 as uuidv4 } from 'uuid'; 

import TypingZoneStatsDisplay from './components/TypingZoneStatsDisplay';
import { useTheme } from './context/ThemeContext';
import { useAuth } from './context/AuthContext';

import Header from './components/Header';
import ResultsDashboard from './components/dashboard/ResultsDashboard';

interface QueuedSnippet {
  id: string;
  words: string[];
  difficulty: number;
}

function App() {
  const { theme, setTheme } = useTheme();
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  
  const [snippetQueue, setSnippetQueue] = useState<QueuedSnippet[]>([]);
  const [recentSnippetIds, setRecentSnippetIds] = useState<string[]>([]);
  const [userState, setUserState] = useState<UserState>(() => {
    return {
      user_id: undefined,
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
  
  // Pause & Timer State
  const [isPaused, setIsPaused] = useState(false);
  const [sessionStartedAt, setSessionStartedAt] = useState<Date | null>(null);
  const [pauseStartTime, setPauseStartTime] = useState<number | null>(null);
  const [totalPausedDuration, setTotalPausedDuration] = useState(0);
  
  // State to hold the full session response from the backend
  const [sessionResultData, setSessionResultData] = useState<SessionResponse | null>(null);

  const [sessionSummary, setSessionSummary] = useState({
    wpm: 0,
    rawWpm: 0,
    accuracy: 0,
    errors: 0,
    duration: 0, 
    keystrokeEvents: [] as KeystrokeEvent[],
    text: "",
    totalWords: 0
  });

  const [liveStats, setLiveStats] = useState({ wpm: 0, accuracy: 100, time: 0 });

  const handleStatsUpdate = useCallback((stats: { wpm: number; accuracy: number; time: number }) => {
    setLiveStats({ wpm: stats.wpm, accuracy: stats.accuracy, time: stats.time });
  }, []);

  const isFetching = useRef(false);

  useEffect(() => {
    if (!authLoading) {
      if (isAuthenticated && user?.id) {
        setUserState(prev => ({ ...prev, user_id: user.id }));
        localStorage.removeItem('flowtype_anonymous_user_id');
      } else {
        let anonymousUserId = localStorage.getItem('flowtype_anonymous_user_id');
        if (!anonymousUserId) {
          anonymousUserId = uuidv4();
          localStorage.setItem('flowtype_anonymous_user_id', anonymousUserId);
        }
        setUserState(prev => ({ ...prev, user_id: anonymousUserId }));
      }
    }
  }, [isAuthenticated, user?.id, authLoading]);


  const fetchMoreSnippets = async (state: UserState, count: number = 1, overrideRecentIds?: string[]) => {
    if (isFetching.current || state.user_id === null) return;
    isFetching.current = true;
    try {
      const fetchedInBatch: string[] = [];
      for (let i = 0; i < count; i++) {
        const currentRecentIds = (overrideRecentIds || recentSnippetIds).filter(Boolean); // Filter falsy values
        const excludeIds = [...currentRecentIds, ...fetchedInBatch];
        const stateWithRecent = { ...state, recentSnippetIds: excludeIds };
        
        const currentId = snippetQueue.length > 0 ? snippetQueue[0]?.id : undefined;
        const snippet = await fetchNextSnippet(stateWithRecent, currentId);
        if (!snippet) {
          break; 
        }
        setSnippetQueue(prev => [...prev, snippet]);
        if (snippet.id) {
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
    if (!authLoading && userState.user_id) {
        fetchMoreSnippets(userState, 2);
    }
  }, [authLoading, userState.user_id]);

  useEffect(() => {
    if (!authLoading && userState.user_id && snippetQueue.length <= 1) {
      fetchMoreSnippets(userState, 1);
    }
  }, [snippetQueue.length, authLoading, userState.user_id]);

  const handleSnippetComplete = async (stats: { // Made async
    wpm: number; 
    accuracy: number; 
    errors: number; 
    duration: number; 
    keystrokeEvents: KeystrokeEvent[]; 
    startedAt: Date 
  }) => {
    if (!sessionStartedAt) {
        setSessionStartedAt(stats.startedAt);
    }

    const currentSnippet = snippetQueue[0];
    if (!currentSnippet) return;

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
    const updatedAllKeystrokes = [...allKeystrokes, ...stats.keystrokeEvents];
    setAllKeystrokes(updatedAllKeystrokes);
    
    // Aggregate session stats
    const now = new Date();
    const startTime = sessionStartedAt || stats.startedAt;
    
    // Correct duration calculation: Total Elapsed - Total Paused Time
    const totalElapsed = now.getTime() - startTime.getTime();
    const totalDuration = Math.max(0, (totalElapsed - totalPausedDuration) / 1000);
    
    const correctKeystrokes = updatedAllKeystrokes.filter(k => k.isCorrect && !k.isBackspace).length;
    const currentSessionWpm = totalDuration > 0 ? (correctKeystrokes / 5) / (totalDuration / 60) : 0;
    
    const totalKeystrokes = updatedAllKeystrokes.length;
    const currentSessionRawWpm = totalDuration > 0 ? (totalKeystrokes / 5) / (totalDuration / 60) : 0;
    
    const currentSessionAccuracy = updatedAllKeystrokes.length > 0 ? (updatedAllKeystrokes.filter(k => k.isCorrect).length / updatedAllKeystrokes.length) : 0;

    const snippetText = currentSnippet.words.join(' ');
    const newText = sessionSummary.text ? sessionSummary.text + " " + snippetText : snippetText;
    const newTotalWords = sessionSummary.totalWords + currentSnippet.words.length;

    setSessionSummary({
      wpm: currentSessionWpm,
      rawWpm: currentSessionRawWpm,
      accuracy: currentSessionAccuracy,
      errors: sessionSummary.errors + stats.errors,
      duration: totalDuration,
      keystrokeEvents: updatedAllKeystrokes,
      text: newText,
      totalWords: newTotalWords
    });

    // Update User State
    const newUserState = {
      ...userState,
      rollingWpm: stats.wpm,
      rollingAccuracy: stats.accuracy,
      currentDifficulty: Math.max(1, Math.min(10, userState.currentDifficulty + (stats.accuracy > 0.9 ? 0.5 : -0.5))),
    };
    setUserState(newUserState);
    
    const updatedRecentIds = [...recentSnippetIds, currentSnippet.id].filter(Boolean).slice(-20); // Filter falsy values
    setRecentSnippetIds(updatedRecentIds);

    setSnippetQueue(prev => prev.slice(1));
    fetchMoreSnippets(newUserState, 1, updatedRecentIds);

    // Save Session to Backend
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
        durationSeconds: totalDuration,
        wordsTyped: newTotalWords,
        keystrokeData: updatedAllKeystrokes,
        wpm: currentSessionWpm,
        accuracy: currentSessionAccuracy,
        errors: sessionSummary.errors + stats.errors,
        difficultyLevel: safeDifficulty,
        snippets: [snippetResult],
        user_state: userState, 
        flowScore: 0.0 
    };

    try {
        const response = await saveSession(sessionPayload);
        setSessionResultData(response); // Store the full response
    } catch (err) {
        console.error("Failed to save session:", err);
    }
  };

  const resetSession = () => {
    setSnippetLogs([]);
    setAllKeystrokes([]);
    setSessionStartedAt(null);
    
    // Reset Pause State
    setIsPaused(false);
    setPauseStartTime(null);
    setTotalPausedDuration(0);
    
    setSessionSummary({
      wpm: 0, rawWpm: 0, accuracy: 0, errors: 0, duration: 0, keystrokeEvents: [], text: "", totalWords: 0
    });
    setUserState(prev => ({
      ...prev,
      rollingWpm: 0,
      rollingAccuracy: 1,
      backspaceRate: 0,
      hesitationCount: 0,
      recentErrors: [],
      currentDifficulty: 5,
    }));
  };

  const handlePause = () => {
      setIsPaused(true);
      setPauseStartTime(Date.now());
  };

  // This now serves as the "Continue" action
  const handleContinue = () => {
    // 1. Reset Session Data (Clear history, logs, timers)
    resetSession();
    setSessionResultData(null); // Clear the session result data to hide dashboard
    
    // 2. Remove the completed snippet from the queue (already handled in handleSnippetComplete)
    // setSnippetQueue(prev => prev.slice(1)); 
    
    // 3. Ensure we have enough snippets
    fetchMoreSnippets(userState, 2);
    
    // 4. Unpause (happens in resetSession too but ensuring logic)
    // setIsPaused(false); // No need, resetSession already does this
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-bg text-text flex items-center justify-center">
        <div className="text-xl font-mono animate-pulse">Loading FlowType...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg text-text flex flex-col p-8 relative font-mono transition-colors duration-300 overflow-hidden">
      
      {/* Header Component (Top Right Controls) */}
      <Header isPaused={isPaused} />

      {/* Main Content Area */}
      <div className="flex-grow flex items-center justify-center w-full max-w-[1800px] mx-auto relative">
         {isPaused && sessionResultData ? ( // Conditionally render if data is available
            <ResultsDashboard 
              sessionResult={sessionResultData} // Pass the full sessionResultData
              onContinue={handleContinue}
            />
         ) : (
            <div className="flex flex-col items-center w-full">
                {/* Dynamic Header Area (Above Snippet) */}
                <div className="w-full mb-8 h-16 relative">
                    <div className="absolute bottom-0 left-0 w-full flex justify-between items-end transition-all duration-500 opacity-100 translate-y-0">
                        <h1 className="text-3xl font-bold text-subtle tracking-tighter">nerdtype</h1>
                        {snippetQueue.length > 0 && (
                            <TypingZoneStatsDisplay wpm={liveStats.wpm} accuracy={liveStats.accuracy} time={liveStats.time} />
                        )}
                    </div>
                </div>

                {/* Typing Zone */}
                <div className="w-full transition-all duration-500 scale-100">
                    {snippetQueue.length > 0 ? (
                        <TypingZone 
                        snippets={snippetQueue} 
                        onSnippetComplete={handleSnippetComplete}
                        onRequestPause={handlePause}
                        onStatsUpdate={handleStatsUpdate}
                        />
                    ) : (
                        <div className="text-center text-subtle animate-pulse">loading...</div>
                    )}
                </div>
            </div>
         )}
      </div>

      {/* Footer Hint (Only when typing) */}
      {!isPaused && (
        <div className="absolute bottom-8 left-0 right-0 text-center text-subtle text-sm transition-opacity duration-300 opacity-100">
              <span className="bg-container px-2 py-1 rounded text-xs mr-2 text-text">enter</span> to pause
        </div>
      )}
    </div>
  );
}

export default App;