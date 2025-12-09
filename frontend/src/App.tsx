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
    };
  });
  const [snippetLogs, setSnippetLogs] = useState<SnippetLog[]>([]);
  const [allKeystrokes, setAllKeystrokes] = useState<KeystrokeEvent[]>([]);
  
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

  const [liveStats, setLiveStats] = useState({ wpm: 0, accuracy: 100, time: 0, timeRemaining: null as number | null, sessionMode: 'free' as '15' | '30' | '60' | '120' | 'free', sessionStarted: false });

  const handleStatsUpdate = useCallback((stats: { wpm: number; accuracy: number; time: number; timeRemaining: number | null; sessionMode: '15' | '30' | '60' | '120' | 'free'; sessionStarted: boolean }) => {
    setLiveStats({ wpm: stats.wpm, accuracy: stats.accuracy, time: stats.time, timeRemaining: stats.timeRemaining, sessionMode: stats.sessionMode, sessionStarted: stats.sessionStarted });
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
    keystrokeEvents: KeystrokeEvent[]; 
  }) => {
    const currentSnippet = snippetQueue[0];
    if (!currentSnippet) return;

    const safeDifficulty = currentSnippet.difficulty ?? 5.0;

    // Create log (timestamps come from keystroke events)
    const log: SnippetLog = {
        snippet_id: currentSnippet.id,
        started_at: new Date(stats.keystrokeEvents[0]?.timestamp || Date.now()).toISOString(),
        completed_at: new Date(stats.keystrokeEvents[stats.keystrokeEvents.length - 1]?.timestamp || Date.now()).toISOString(),
        wpm: 0, // Will be calculated in backend
        accuracy: 0, // Will be calculated in backend
        difficulty: safeDifficulty
    };

    setSnippetLogs(prev => [...prev, log]);
    const updatedAllKeystrokes = [...allKeystrokes, ...stats.keystrokeEvents];
    setAllKeystrokes(updatedAllKeystrokes);
    
    // Calculate stats from all keystrokes for display purposes only
    const correctKeystrokes = updatedAllKeystrokes.filter(k => k.isCorrect && !k.isBackspace).length;
    const totalErrors = updatedAllKeystrokes.filter(k => !k.isCorrect && !k.isBackspace).length;
    
    // Calculate duration from keystroke timestamps
    const sortedKeystrokes = [...updatedAllKeystrokes].sort((a, b) => a.timestamp - b.timestamp);
    const duration_ms = sortedKeystrokes[sortedKeystrokes.length - 1].timestamp - sortedKeystrokes[0].timestamp;
    const duration_seconds = duration_ms / 1000;
    
    const currentSessionWpm = duration_seconds > 0 ? (correctKeystrokes / 5) / (duration_seconds / 60) : 0;
    const totalKeystrokes = updatedAllKeystrokes.filter(k => !k.isBackspace).length;
    const currentSessionRawWpm = duration_seconds > 0 ? (totalKeystrokes / 5) / (duration_seconds / 60) : 0;
    const currentSessionAccuracy = totalKeystrokes > 0 ? (correctKeystrokes / totalKeystrokes) : 0;

    const snippetText = currentSnippet.words.join(' ');
    const newText = sessionSummary.text ? sessionSummary.text + " " + snippetText : snippetText;
    const newTotalWords = sessionSummary.totalWords + currentSnippet.words.length;

    setSessionSummary({
      wpm: currentSessionWpm,
      rawWpm: currentSessionRawWpm,
      accuracy: currentSessionAccuracy,
      errors: totalErrors,
      duration: duration_seconds,
      keystrokeEvents: updatedAllKeystrokes,
      text: newText,
      totalWords: newTotalWords
    });

    // Update User State
    const newUserState = {
      ...userState,
      rollingWpm: currentSessionWpm,
      rollingAccuracy: currentSessionAccuracy,
    };
    setUserState(newUserState);
    
    const updatedRecentIds = [...recentSnippetIds, currentSnippet.id].filter(Boolean).slice(-20);
    setRecentSnippetIds(updatedRecentIds);

    // Move to next snippet (don't save session yet - wait for Enter key)
    setSnippetQueue(prev => prev.slice(1));
    fetchMoreSnippets(newUserState, 1, updatedRecentIds);
  };

  const handlePause = async () => {
      // User pressed Enter - end session and save to backend
      if (allKeystrokes.length === 0) return; // No keystrokes to save

      const sortedKeystrokes = [...allKeystrokes].sort((a, b) => a.timestamp - b.timestamp);
      const duration_ms = sortedKeystrokes[sortedKeystrokes.length - 1].timestamp - sortedKeystrokes[0].timestamp;
      const duration_seconds = duration_ms / 1000;

      // Build snippet results from logs
      const snippetResults: SnippetResult[] = snippetLogs.map(log => ({
        snippet_id: log.snippet_id,
        wpm: 0, // Backend will recalculate
        accuracy: 0, // Backend will recalculate
        difficulty: log.difficulty,
        started_at: new Date(log.started_at).getTime(),
        completed_at: new Date(log.completed_at).getTime()
      }));

      const sessionPayload: SessionCreateRequest = {
          user_id: userState.user_id,
          durationSeconds: duration_seconds,
          wordsTyped: sessionSummary.totalWords,
          keystrokeData: allKeystrokes,
          difficultyLevel: snippetLogs.length > 0 ? snippetLogs[0].difficulty : 5.0,
          snippets: snippetResults,
          user_state: userState, 
          flowScore: 0.0 
      };

      try {
          const response = await saveSession(sessionPayload);
          setSessionResultData(response); // Store the full response - this triggers dashboard view
      } catch (err) {
          console.error("Failed to save session:", err);
      }
  };

  const resetSession = () => {
    setSnippetLogs([]);
    setAllKeystrokes([]);
    
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
    }));
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
      <Header isPaused={!!sessionResultData} sessionStarted={liveStats.sessionStarted} />

      {/* Main Content Area */}
      <div className="flex-grow flex items-center justify-center w-full max-w-[1800px] mx-auto relative">
         {sessionResultData ? ( // Conditionally render if data is available
            <ResultsDashboard 
              sessionResult={sessionResultData} // Pass the full sessionResultData
              onContinue={handleContinue}
            />
         ) : (
            <div className="flex flex-col items-center w-full">
                {/* Dynamic Header Area (Above Snippet) */}
                <div className="w-full mb-8 h-16 relative">
                    <div className="absolute bottom-0 left-0 w-full flex justify-between items-end transition-all duration-500">
                        <h1 className={`text-3xl font-bold text-subtle tracking-tighter ${
                            !liveStats.sessionStarted ? 'opacity-0 pointer-events-none' : 'opacity-100 transition-opacity duration-500'
                        }`}>nerdtype</h1>
                        {snippetQueue.length > 0 && (
                            <TypingZoneStatsDisplay 
                                wpm={liveStats.wpm} 
                                accuracy={liveStats.accuracy} 
                                time={liveStats.time}
                                sessionMode={liveStats.sessionMode}
                                timeRemaining={liveStats.timeRemaining}
                            />
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
      {!sessionResultData && (
        <div className="absolute bottom-8 left-0 right-0 text-center text-subtle text-sm transition-opacity duration-300 opacity-100">
              <span className="bg-container px-2 py-1 rounded text-xs mr-2 text-text">enter</span> to pause
        </div>
      )}
    </div>
  );
}

export default App;