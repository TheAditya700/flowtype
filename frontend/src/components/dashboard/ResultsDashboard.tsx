import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { fetchUserProfile } from '../../api/client';
import { UserProfile, SessionResponse } from '../../types';
import { Play } from 'lucide-react';

// New Widgets
import SpeedGraph from './SpeedGraph';
import FlowRadar from './FlowRadar';
import IkiHistogramWidget from './IkiHistogramWidget';
import SessionStatsWidget from './SessionStatsWidget';
import KeyboardHeatmap from './KeyboardHeatmap';
import ReplayChunkStrip from './ReplayChunkStrip';
import SkillBars from './SkillBars';
import RolloverBreakdown from './RolloverBreakdown';

interface ResultsDashboardProps {
  sessionResult: SessionResponse;
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
  sessionResult,
  onContinue
}) => {
  const { isAuthenticated, user } = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [previousStats, setPreviousStats] = useState<StatsData | undefined>(undefined);

  // All data is now directly in sessionResult (no nested analytics)
  const { wpm, accuracy, rawWpm } = sessionResult;

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
    // --- Previous Stats Logic ---
    const stored = localStorage.getItem('flowtype_last_run_stats');
    if (stored) {
        try {
            setPreviousStats(JSON.parse(stored));
        } catch (e) {
            console.warn("Failed to parse previous stats", e);
        }
    }
    
    // Construct current stats object for comparison and next 'previous'
    const currentStats: StatsData = {
        wpm: sessionResult.wpm,
        rawWpm: sessionResult.wpm, // Assuming wpm in sessionResult is effective WPM, rawWpm not explicitly passed
        time: sessionResult.durationSeconds,
        accuracy: sessionResult.accuracy,
        avgChunkLength: sessionResult.avgChunkLength,
        errors: sessionResult.errors,
        kspc: sessionResult.kspc,
        avgIki: sessionResult.avgIki,
        rolloverRate: sessionResult.rollover / 100
    };
    
    // Save current as new previous for NEXT time
    localStorage.setItem('flowtype_last_run_stats', JSON.stringify(currentStats));

  }, [sessionResult]); // Dependencies updated

  // No longer need to check if analytics is null, as it's directly from props
  // and type-guaranteed by SessionResponse
  
  return (
    <div className="p-6 max-w-[1800px] mx-auto flex flex-col gap-6">
      
      {/* Grid Layout: 3 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* --- Left Column: Skill Bars (Spans 2 rows) --- */}
        <div className="lg:row-span-2">
            <SkillBars 
                accuracy={sessionResult.accuracy} 
                consistency={sessionResult.smoothness} 
                speed={sessionResult.wpm}
                durationSeconds={sessionResult.durationSeconds}
            />
        </div>

        {/* --- Top Right Section --- */}
        {/* Col 2-3: Speed Graph (Spans 2 cols) */}
        <div className="lg:col-span-2">
          <SpeedGraph data={sessionResult.speedSeries} />
        </div>

        {/* --- Flow Radar (Row 2, left side of right section) --- */}
       

        {/* --- Replay Strip (Spans 2 cols, row 2 right side) --- */}
        <div className="lg:col-span-2">
            <ReplayChunkStrip events={sessionResult.replayEvents} />
        </div>

        {/* --- Keyboard Heatmap (Spans 2 cols, row 3) --- */}
        <div className="lg:col-span-2">
          <KeyboardHeatmap charStats={sessionResult.heatmapData} />
        </div>

        {/* --- Flow Radar (Row 3, single col, right of Keyboard Heatmap) --- */}
        <div>
            <FlowRadar 
                metrics={{
                    smoothness: sessionResult.smoothness,
                    rollover: sessionResult.rollover,
                    leftFluency: sessionResult.leftFluency,
                    rightFluency: sessionResult.rightFluency,
                    crossFluency: sessionResult.crossFluency
                }}
            />
        </div>

        {/* --- IKI Histogram (Row 4, single col) --- */}
        <div>
            <IkiHistogramWidget 
                replayEvents={sessionResult.replayEvents}
            />
        </div>

        {/* --- Rollover Breakdown (Row 4, col-span-2) --- */}
        <div className="lg:col-span-2">
            <RolloverBreakdown 
                rolloverL2L={sessionResult.rolloverL2L}
                rolloverR2R={sessionResult.rolloverR2R}
                rolloverCross={sessionResult.rolloverCross}
                rollover={sessionResult.rollover}
            />
        </div>

        {/* --- Session Stats (Row 5, full width) --- */}
        <div className="lg:col-span-3">
            <SessionStatsWidget 
                errors={sessionResult.errors}
                rawWpm={sessionResult.rawWpm}
                kspc={sessionResult.kspc}
                avgChunkLength={sessionResult.avgChunkLength}
                ikiStdDev={sessionResult.avgIki}
                smoothness={sessionResult.smoothness}
            />
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