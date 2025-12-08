import React from 'react';
import { ArrowUp, ArrowDown, Minus, Activity, Trophy, Hash, Clock, Zap } from 'lucide-react';

interface LifetimeStatItemProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}

const LifetimeStatItem: React.FC<LifetimeStatItemProps> = ({ label, value, icon }) => {
  return (
    <div className="bg-gray-800/30 p-2 rounded-lg border border-gray-700/50 flex items-center gap-3">
      <div className="p-1.5 bg-gray-800 rounded-md text-gray-400">
        {icon}
      </div>
      <div>
        <div className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">{label}</div>
        <div className="text-sm font-bold text-white">{value}</div>
      </div>
    </div>
  );
};

interface LifetimeStatsBoxProps {
  stats: {
    total_sessions: number;
    avg_wpm: number;
    avg_accuracy: number;
    total_time_typing: number;
    best_wpm_15: number;
    best_wpm_30: number;
    best_wpm_60: number;
    best_wpm_120: number;
  };
}

const LifetimeStatsBox: React.FC<LifetimeStatsBoxProps> = ({ stats }) => {
  const formatDuration = (seconds: number) => {
      if (!seconds) return '0s';
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      if (h > 0) return `${h}h ${m}m`;
      if (m > 0) return `${m}m ${s}s`;
      return `${s}s`;
  };

  return (
    <div className="w-full bg-gray-900 rounded-xl p-4 border border-gray-800 flex flex-col h-full relative">
      <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-3">Lifetime Stats</h3>
      
      <div className="grid grid-cols-2 gap-2 pt-2 overflow-y-auto">
        {/* General */}
        <LifetimeStatItem 
            label="Snippets" 
            value={stats.total_sessions} 
            icon={<Hash size={14} className="text-purple-400"/>} 
        />
        <LifetimeStatItem 
            label="Time Typed" 
            value={formatDuration(stats.total_time_typing)} 
            icon={<Clock size={14} className="text-blue-400"/>} 
        />
        <LifetimeStatItem 
            label="Avg WPM" 
            value={Math.round(stats.avg_wpm)} 
            icon={<Activity size={14} className="text-blue-400"/>} 
        />
        <LifetimeStatItem 
            label="Avg Acc" 
            value={`${Math.round(stats.avg_accuracy * 100)}%`} 
            icon={<Trophy size={14} className="text-emerald-400"/>} 
        />

        {/* Bests */}
        <LifetimeStatItem 
            label="Best 15s" 
            value={Math.round(stats.best_wpm_15)} 
            icon={<Zap size={14} className="text-yellow-400"/>} 
        />
        <LifetimeStatItem 
            label="Best 30s" 
            value={Math.round(stats.best_wpm_30)} 
            icon={<Zap size={14} className="text-yellow-400"/>} 
        />
        <LifetimeStatItem 
            label="Best 60s" 
            value={Math.round(stats.best_wpm_60)} 
            icon={<Zap size={14} className="text-yellow-400"/>} 
        />
        <LifetimeStatItem 
            label="Best 120s" 
            value={Math.round(stats.best_wpm_120)} 
            icon={<Zap size={14} className="text-yellow-400"/>} 
        />
      </div>
    </div>
  );
};

export default LifetimeStatsBox;
