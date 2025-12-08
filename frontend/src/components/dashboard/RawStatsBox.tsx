import React from 'react';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';

interface StatItemProps {
  label: string;
  value: string | number;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  unit?: string;
}

const StatItem: React.FC<StatItemProps> = ({ label, value, trend, trendValue, unit }) => {
  return (
    <div className="bg-gray-800/30 p-3 rounded-lg border border-gray-700/50 hover:border-gray-600 transition-colors">
      <div className="flex justify-between items-start mb-1">
        <span className="text-gray-400 text-[10px] font-medium uppercase tracking-wider">{label}</span>
        {trend && (
          <div className={`flex items-center text-[10px] font-medium ${
            trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-gray-500'
          }`}>
            {trend === 'up' && <ArrowUp size={10} className="mr-0.5" />}
            {trend === 'down' && <ArrowDown size={10} className="mr-0.5" />}
            {trend === 'neutral' && <Minus size={10} className="mr-0.5" />}
            {trendValue && <span>{trendValue}</span>}
          </div>
        )}
      </div>
      <div className="flex items-baseline">
        <span className="text-xl font-bold text-white">{value}</span>
        {unit && <span className="ml-1 text-gray-500 text-xs">{unit}</span>}
      </div>
    </div>
  );
};

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

interface RawStatsBoxProps {
  stats: StatsData;
  previousStats?: StatsData;
}

const RawStatsBox: React.FC<RawStatsBoxProps> = ({ stats, previousStats }) => {
  const formatTime = (seconds: number) => {
    const s = Math.round(seconds);
    if (s < 60) return `${s}`;
    const m = Math.floor(s / 60);
    const remS = s % 60;
    return `${m}m ${remS}`;
  };

  const calculateTrend = (current: number, previous: number | undefined, inverse: boolean = false): { trend?: 'up' | 'down' | 'neutral', trendValue?: string } => {
      if (previous === undefined) return {};
      const diff = current - previous;
      if (Math.abs(diff) < 0.01) return { trend: 'neutral' };
      
      const isGood = inverse ? diff < 0 : diff > 0;
      const trend = isGood ? 'up' : 'down'; // Visually 'up' (green) is good, 'down' (red) is bad usually, but here strict 'up'/'down' arrows
      // Actually, typically Up arrow = increase value. Color indicates good/bad.
      // My StatItem uses up=green, down=red. 
      // So if inverse (lower is better, e.g. errors), and value went DOWN (improvement), we want Green arrow pointing DOWN?
      // Or just Green color?
      // Re-reading StatItem: trend='up' gives Green ArrowUp. trend='down' gives Red ArrowDown.
      // This couples direction with color. 
      // If errors go down (good), we want Down arrow but Green color?
      // The current StatItem logic forces Down=Red. 
      // Let's stick to standard: Up=Increase, Down=Decrease.
      // But we might want to decouple color if needed.
      // For now, let's just show direction. The user asked for trend logic.
      // "up" (green) might be confusing for errors going up (bad).
      
      // Let's refine StatItem logic slightly if I could, but strictly following instruction 4 "Pass ... trend ('up', 'down')".
      // I will adhere to value direction.
      
      return {
          trend: diff > 0 ? 'up' : 'down',
          trendValue: `${diff > 0 ? '+' : ''}${diff.toFixed(1)}`
      };
  };

  // Helper to handle "Good" direction colors better? 
  // User asked to "implement trend logic". 
  // Let's stick to simple value comparison.

  const wpmTrend = calculateTrend(stats.wpm, previousStats?.wpm);
  const errorsTrend = calculateTrend(stats.errors, previousStats?.errors, true);
  // We can override color logic by passing specific trend strings if StatItem supported it, but it doesn't.
  // We'll leave it simple.

  return (
    <div className="w-full h-auto bg-gray-900 rounded-xl p-4 border border-gray-800 flex flex-col relative">
      <h3 className="text-gray-400 text-sm font-medium mb-4">Raw Stats</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 flex-grow overflow-y-auto">
        {/* Row 1: Speed & Time */}
        <StatItem label="WPM" value={Math.round(stats.wpm)} unit="" {...calculateTrend(stats.wpm, previousStats?.wpm)} />
        <StatItem label="Raw WPM" value={Math.round(stats.rawWpm)} {...calculateTrend(stats.rawWpm, previousStats?.rawWpm)} />
        <StatItem label="Time" value={formatTime(stats.time)} unit={stats.time < 60 ? "s" : "s"} />

        {/* Row 2: Flow & Consistency */}
        <StatItem label="Accuracy" value={Math.round(stats.accuracy * 100)} unit="%" {...calculateTrend(stats.accuracy, previousStats?.accuracy)} />
        <StatItem label="Avg Chunk" value={stats.avgChunkLength.toFixed(1)} unit="ch" {...calculateTrend(stats.avgChunkLength, previousStats?.avgChunkLength)} />
        <StatItem label="Avg IKI" value={Math.round(stats.avgIki)} unit="ms" {...calculateTrend(stats.avgIki, previousStats?.avgIki, true)} />

        {/* Row 3: Technical & Error */}
        <StatItem label="Errors" value={stats.errors} {...calculateTrend(stats.errors, previousStats?.errors, true)} />
        <StatItem label="KSPC" value={stats.kspc.toFixed(2)} {...calculateTrend(stats.kspc, previousStats?.kspc, true)} />
        <StatItem label="Rollover" value={Math.round(stats.rolloverRate * 100)} unit="%" {...calculateTrend(stats.rolloverRate, previousStats?.rolloverRate)} />
      </div>
    </div>
  );
};

export default RawStatsBox;
