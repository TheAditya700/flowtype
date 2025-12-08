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
    <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
      <div className="flex justify-between items-start mb-1">
        <span className="text-gray-400 text-xs font-medium uppercase tracking-wider">{label}</span>
        {trend && (
          <div className={`flex items-center text-xs ${
            trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-gray-500'
          }`}>
            {trend === 'up' && <ArrowUp size={12} className="mr-0.5" />}
            {trend === 'down' && <ArrowDown size={12} className="mr-0.5" />}
            {trend === 'neutral' && <Minus size={12} className="mr-0.5" />}
            {trendValue && <span>{trendValue}</span>}
          </div>
        )}
      </div>
      <div className="flex items-baseline">
        <span className="text-2xl font-bold text-white">{value}</span>
        {unit && <span className="ml-1 text-gray-500 text-sm">{unit}</span>}
      </div>
    </div>
  );
};

interface RawStatsBoxProps {
  stats: {
    wpm: number;
    accuracy: number;
    kspc: number;
    errors: number;
    avgIki: number;
    rolloverRate: number;
  };
}

const RawStatsBox: React.FC<RawStatsBoxProps> = ({ stats }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 w-full">
      <StatItem label="WPM" value={Math.round(stats.wpm)} unit="" trend="up" trendValue="2%" />
      <StatItem label="Accuracy" value={Math.round(stats.accuracy * 100)} unit="%" />
      <StatItem label="Errors" value={stats.errors} trend="down" trendValue="-1" />
      <StatItem label="KSPC" value={stats.kspc.toFixed(2)} />
      <StatItem label="Avg IKI" value={Math.round(stats.avgIki)} unit="ms" />
      <StatItem label="Rollover" value={Math.round(stats.rolloverRate * 100)} unit="%" />
    </div>
  );
};

export default RawStatsBox;
