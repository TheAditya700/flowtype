import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { fetchUserProfile, calculateSessionMetrics } from '../../api/client';
import { UserProfile, KeystrokeEvent, AnalyticsResponse, SnippetLog } from '../../types';
import { Play } from 'lucide-react';

// New Widgets
import SpeedGraph from './SpeedGraph';
import FlowRadar from './FlowRadar';
import RawStatsBox from './RawStatsBox';
import KeyboardHeatmap from './KeyboardHeatmap';
import ReplayChunkStrip from './ReplayChunkStrip';
import SkillBars from './SkillBars';
import LifetimeStatsBox from './LifetimeStatsBox';
import ConfidenceWidget from './ConfidenceWidget';

interface ResultsDashboardProps {
  keystrokeEvents: KeystrokeEvent[];
  wpm: number;
  rawWpm: number;
  accuracy: number;
  duration: number;
  snippetText: string;
  snippetLogs?: SnippetLog[]; 
  onContinue: () => void;
}

interface StatsData {
    wpm: number;
    rawWpm: number;
    time: number;
    accuracy: number;
    avgChunkLength: number;
    errors: number;
    kspc: number;
    avgIki: number;
    rolloverRate: number;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  keystrokeEvents,
  wpm,
  rawWpm,
  accuracy,
  duration,
  snippetText,
  snippetLogs = [],
  onContinue
}) => {
  const { isAuthenticated, user } = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [sessionAnalytics, setSessionAnalytics] = useState<AnalyticsResponse | null>(null);
  const [previousStats, setPreviousStats] = useState<StatsData | undefined>(undefined);

  useEffect(() => {
    const loadProfile = async () => {
      if (isAuthenticated && user?.id) {
        try {
          const profile = await fetchUserProfile();
          setUserProfile(profile);
        } catch (err) {
          console.error("Failed to load profile", err);
        }
      }
    };
    loadProfile();
  }, [isAuthenticated, user?.id]);

  useEffect(() => {
    const fetchAnalytics = async () => {
        try {
            // Map logs to boundaries
            const snippetBoundaries = snippetLogs.map(log => ({
                startTime: new Date(log.started_at).getTime(),
                endTime: new Date(log.completed_at).getTime()
            }));

            const result = await calculateSessionMetrics({
                keystrokeData: keystrokeEvents,
                wpm,
                accuracy,
                snippetBoundaries: snippetBoundaries.length > 0 ? snippetBoundaries : undefined
            });
            setSessionAnalytics(result);
            
            // Handle Previous Stats Logic
            const stored = localStorage.getItem('flowtype_last_run_stats');
            if (stored) {
                try {
                    setPreviousStats(JSON.parse(stored));
                } catch (e) {
                    console.warn("Failed to parse previous stats", e);
                }
            }
            
            // Construct current stats object
            const currentStats: StatsData = {
                wpm,
                rawWpm,
                time: duration,
                accuracy,
                avgChunkLength: result.avgChunkLength,
                errors: result.errors,
                kspc: result.kspc,
                avgIki: result.avgIki,
                rolloverRate: result.rollover / 100
            };
            
            // Save current as new previous for NEXT time
            localStorage.setItem('flowtype_last_run_stats', JSON.stringify(currentStats));

        } catch (error) {
            console.error("Failed to fetch session analytics", error);
        }
    };
    
    if (keystrokeEvents.length > 0) {
        fetchAnalytics();
    }
  }, [keystrokeEvents, wpm, accuracy, snippetLogs, rawWpm, duration]);

  if (!sessionAnalytics) {
    return (
        <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-xl font-mono animate-pulse text-subtle">Analyzing flow...</div>
        </div>
    );
  }

  return (
    <div className="p-6 max-w-[1800px] mx-auto flex flex-col gap-6">
      
      {/* Grid Layout: 4 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* --- Row 1 --- */}
        {/* Col 1: Skill Bars */}
        <div className="lg:col-span-1">
            <SkillBars 
                accuracy={accuracy} 
                consistency={sessionAnalytics.smoothness} 
                speed={wpm} 
            />
        </div>
        {/* Col 2-3: Speed Graph */}
        <div className="lg:col-span-2">
          <SpeedGraph data={sessionAnalytics.speedSeries} />
        </div>
        {/* Col 4: Flow Radar */}
        <div className="lg:col-span-1">
          <FlowRadar metrics={{
              smoothness: sessionAnalytics.smoothness,
              rollover: sessionAnalytics.rollover,
              leftFluency: sessionAnalytics.leftFluency,
              rightFluency: sessionAnalytics.rightFluency,
              crossFluency: sessionAnalytics.crossFluency
          }} />
        </div>


        {/* --- Row 2 --- */}
        {/* Col 1: Lifetime Stats */}
        <div className="lg:col-span-1">
            <LifetimeStatsBox 
                stats={userProfile?.stats || { 
                    total_sessions: 0, 
                    avg_wpm: 0, 
                    avg_accuracy: 0,
                    total_time_typing: 0,
                    best_wpm_15: 0,
                    best_wpm_30: 0,
                    best_wpm_60: 0,
                    best_wpm_120: 0
                }}
            />
        </div>
        {/* Col 2: Raw Stats */}
        <div className="lg:col-span-1">
          <RawStatsBox 
            stats={{
              wpm,
              rawWpm,
              time: duration,
              accuracy,
              avgChunkLength: sessionAnalytics.avgChunkLength,
              errors: sessionAnalytics.errors,
              kspc: sessionAnalytics.kspc,
              avgIki: sessionAnalytics.avgIki,
              rolloverRate: sessionAnalytics.rollover / 100
            }} 
            previousStats={previousStats}
          />
        </div>
        {/* Col 3-4: Heatmap */}
        <div className="lg:col-span-2">
          <KeyboardHeatmap charStats={sessionAnalytics.heatmapData} />
        </div>


        {/* --- Row 3 --- */}
        {/* Col 1: Confidence Widget */}
        <div className="lg:col-span-1">
            <ConfidenceWidget 
                left={sessionAnalytics.leftFluency} 
                right={sessionAnalytics.rightFluency} 
                cross={sessionAnalytics.crossFluency} 
            />
        </div>
        {/* Col 2-4: Replay Strip */}
        <div className="lg:col-span-3">
            <ReplayChunkStrip events={sessionAnalytics.replayEvents} />
        </div>

      </div>

      {/* Controls */}
      <div className="flex justify-center mt-4">
        <button 
          onClick={onContinue}
          className="flex items-center gap-2 px-12 py-4 bg-blue-600 text-white hover:bg-blue-500 transition font-bold shadow-lg shadow-blue-500/20 rounded-xl text-lg tracking-wide"
        >
          <Play size={20} fill="currentColor" /> Continue
        </button>
      </div>
    </div>
  );
};

export default ResultsDashboard;