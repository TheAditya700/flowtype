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

interface ResultsDashboardProps {
  keystrokeEvents: KeystrokeEvent[];
  wpm: number;
  accuracy: number;
  duration: number;
  snippetText: string;
  snippetLogs?: SnippetLog[]; 
  onContinue: () => void;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  keystrokeEvents,
  wpm,
  accuracy,
  duration,
  snippetText,
  snippetLogs = [],
  onContinue
}) => {
  const { isAuthenticated, user } = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [sessionAnalytics, setSessionAnalytics] = useState<AnalyticsResponse | null>(null);

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
        } catch (error) {
            console.error("Failed to fetch session analytics", error);
        }
    };
    
    if (keystrokeEvents.length > 0) {
        fetchAnalytics();
    }
  }, [keystrokeEvents, wpm, accuracy, snippetLogs]);

  if (!sessionAnalytics) {
    return (
        <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-xl font-mono animate-pulse text-subtle">Analyzing flow...</div>
        </div>
    );
  }

  return (
    <div className="p-6 max-w-[1200px] mx-auto flex flex-col gap-6">
      
      {/* Top Row: Speed & Flow */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <SpeedGraph data={sessionAnalytics.speedSeries} />
        </div>
        <div className="lg:col-span-1">
          <FlowRadar metrics={{
              smoothness: sessionAnalytics.smoothness,
              rollover: sessionAnalytics.rollover,
              leftFluency: sessionAnalytics.leftFluency,
              rightFluency: sessionAnalytics.rightFluency,
              crossFluency: sessionAnalytics.crossFluency
          }} />
        </div>
      </div>

      {/* Middle Row: Stats & Heatmap */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <RawStatsBox stats={{
              wpm,
              accuracy,
              errors: sessionAnalytics.errors,
              kspc: sessionAnalytics.kspc,
              avgIki: sessionAnalytics.avgIki,
              rolloverRate: sessionAnalytics.rollover / 100
          }} />
        </div>
        <div className="lg:col-span-2">
          <KeyboardHeatmap charStats={sessionAnalytics.heatmapData} />
        </div>
      </div>

      {/* Bottom Row: Replay & Rollover */}
      <ReplayChunkStrip events={sessionAnalytics.replayEvents} />

      {/* Controls */}
      <div className="flex justify-center mt-8">
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