import { useState, useEffect, useRef, useCallback } from 'react';
import TypingZone from './components/TypingZone';
import { fetchNextSnippet, saveSession } from './api/client';
import { UserState, SnippetLog, KeystrokeEvent, SessionCreateRequest, SnippetResult } from './types';
import { RotateCcw, Play, User, Palette } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid'; 

import TypingZoneStatsDisplay from './components/TypingZoneStatsDisplay';
import Heatmap from './components/Heatmap';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useTheme } from './context/ThemeContext';

import StatsWidget from './components/StatsWidget'; // <-- ADDED
import GraphWidget from './components/GraphWidget';
import BottomControls from './components/BottomControls';
import Header from './components/Header'; // <-- ADDED

interface QueuedSnippet {
  id: string;
  words: string[];
  difficulty: number;
}

function App() {
  const { theme, setTheme } = useTheme(); // Theme Context
  
  const [snippetQueue, setSnippetQueue] = useState<QueuedSnippet[]>([]);
  const [recentSnippetIds, setRecentSnippetIds] = useState<string[]>([]);
  const [userState, setUserState] = useState<UserState>(() => {
    let userId = localStorage.getItem('flowtype_user_id');
    if (!userId) {
      userId = uuidv4();
      localStorage.setItem('flowtype_user_id', userId);
    }
    return {
      user_id: userId, 
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
        const currentRecentIds = overrideRecentIds || recentSnippetIds;
        const excludeIds = [...currentRecentIds, ...fetchedInBatch];
        const stateWithRecent = { ...state, recentSnippetIds: excludeIds };
        
        const currentId = snippetQueue.length > 0 ? snippetQueue[0]?.id : undefined;
        const snippet = await fetchNextSnippet(stateWithRecent, currentId);
        if (!snippet) {
          break; 
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
    
    const updatedRecentIds = [...recentSnippetIds, currentSnippet.id].slice(-20);
    setRecentSnippetIds(updatedRecentIds);

    setSnippetQueue(prev => prev.slice(1));
    fetchMoreSnippets(newUserState, 1, updatedRecentIds);

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
        wordsTyped: currentSnippet.words.length,
        keystrokeData: stats.keystrokeEvents,
        wpm: stats.wpm,
        accuracy: stats.accuracy,
        errors: stats.errors,
        difficultyLevel: safeDifficulty,
        snippets: [snippetResult],
        user_state: userState, 
        flowScore: 0.0 
    };

    saveSession(sessionPayload).catch(err => console.error("Failed to save session:", err));
  };

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
  };

  // Calculation for Graphs/Stats (Memoized or inline)
  const chartData = snippetLogs.map((log, idx) => ({
    name: idx + 1,
    wpm: log.wpm,
    accuracy: (log.accuracy * 100).toFixed(1),
  }));

  const avgWpm = snippetLogs.length > 0 
    ? (snippetLogs.reduce((acc, l) => acc + l.wpm, 0) / snippetLogs.length).toFixed(0) 
    : 0;
  const avgAcc = snippetLogs.length > 0 
    ? (snippetLogs.reduce((acc, l) => acc + l.accuracy, 0) / snippetLogs.length * 100).toFixed(1) 
    : 100;


  return (
    <div className="min-h-screen bg-bg text-text flex flex-col p-8 relative font-mono transition-colors duration-300 overflow-hidden">
      
      {/* Header Component (Top Right Controls) */}
      <Header isPaused={isPaused} />

      {/* Main Grid Layout */}
      <div className="flex-grow grid grid-cols-[1fr_minmax(auto,1200px)_1fr] gap-8 items-center w-full max-w-[1800px] mx-auto relative">
         
         {/* Left Column: Stats Widget */}
         <div className={`transition-all duration-500 ${isPaused ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-20 pointer-events-none'}`}>
            <StatsWidget 
                wpm={avgWpm} 
                accuracy={avgAcc} 
                totalWords={sessionStats.totalWords} 
            />
         </div>

         {/* Center Column: Logo, Stats, Typing Zone */}
         <div className="flex flex-col items-center w-full">
            
            {/* Dynamic Header Area (Above Snippet) */}
            <div className="w-full mb-8 h-16 relative">
                {/* State 1: Typing (Small Logo + Live Stats) */}
                <div className={`absolute bottom-0 left-0 w-full flex justify-between items-end transition-all duration-500 ${isPaused ? 'opacity-0 translate-y-4 pointer-events-none' : 'opacity-100 translate-y-0'}`}>
                    <h1 className="text-3xl font-bold text-subtle tracking-tighter">nerdtype</h1>
                    {snippetQueue.length > 0 && (
                        <TypingZoneStatsDisplay wpm={liveStats.wpm} accuracy={liveStats.accuracy} />
                    )}
                </div>

                {/* State 2: Paused (Big Logo Centered) */}
                <div className={`absolute bottom-0 left-0 w-full flex justify-center items-end transition-all duration-500 ${isPaused ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
                    <h1 className="text-6xl font-bold text-primary tracking-tighter">nerdtype</h1>
                </div>
            </div>

            {/* Typing Zone */}
            <div className={`w-full transition-all duration-500 ${isPaused ? 'blur-[2px] opacity-50 scale-95' : 'scale-100'}`}>
                {snippetQueue.length > 0 ? (
                    <TypingZone 
                    snippets={snippetQueue} 
                    onSnippetComplete={handleSnippetComplete}
                    onRequestPause={() => setIsPaused(true)}
                    onStatsUpdate={handleStatsUpdate}
                    />
                ) : (
                    <div className="text-center text-subtle animate-pulse">loading...</div>
                )}
            </div>
         </div>

         {/* Right Column: Graph Widget */}
         <div className={`transition-all duration-500 ${isPaused ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-20 pointer-events-none'}`}>
            <GraphWidget data={chartData} />
         </div>

      </div>

      {/* Bottom: Controls */}
      <div className={`absolute bottom-8 left-0 right-0 transition-all duration-500 ${isPaused ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0 pointer-events-none'}`}>
         <BottomControls 
            keystrokes={allKeystrokes} 
            onReset={resetSession} 
            onResume={() => setIsPaused(false)} 
         />
      </div>

      {/* Footer Hint (Only when typing) */}
      <div className={`absolute bottom-8 left-0 right-0 text-center text-subtle text-sm transition-opacity duration-300 ${isPaused ? 'opacity-0' : 'opacity-100'}`}>
            <span className="bg-container px-2 py-1 rounded text-xs mr-2 text-text">enter</span> to pause
      </div>

    </div>
  );
}

export default App;
