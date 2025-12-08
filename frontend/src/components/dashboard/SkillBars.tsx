import React from 'react';

interface SkillBarsProps {
  accuracy: number;    // 0-100
  consistency: number; // 0-100
  speed: number;       // WPM
}

const SkillBar: React.FC<{ label: string; value: number; color: string; max?: number; format?: (v: number) => string }> = ({ label, value, color, max = 100, format }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between text-xs font-medium text-gray-400 mb-1 uppercase tracking-wider">
        <span>{label}</span>
        <span className="text-gray-200">{format ? format(value) : Math.round(value)}</span>
      </div>
      <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} transition-all duration-1000 ease-out`} 
          style={{ width: `${percentage}%` }} 
        />
      </div>
    </div>
  );
};

const SkillBars: React.FC<SkillBarsProps> = ({ accuracy, consistency, speed }) => {
  return (
    <div className="w-full bg-gray-900 rounded-xl p-5 border border-gray-800 h-full flex flex-col justify-center">
      <SkillBar 
        label="Accuracy" 
        value={accuracy * 100} 
        color="bg-emerald-500" 
        format={(v) => `${Math.round(v)}%`}
      />
      <SkillBar 
        label="Consistency" 
        value={consistency} 
        color="bg-blue-500" 
        format={(v) => `${Math.round(v)}`}
      />
      <SkillBar 
        label="Speed" 
        value={speed} 
        max={200} // Assuming 200 is a "full bar" goal for visualization
        color="bg-purple-500" 
        format={(v) => `${Math.round(v)} wpm`}
      />
    </div>
  );
};

export default SkillBars;