import React from 'react';

interface SkillBarsProps {
  accuracy: number;      // 0-1 or 0-100
  consistency: number;   // 0-100
  speed: number;         // WPM
  durationSeconds?: number; // Duration in seconds (optional)
}

const SkillBars: React.FC<SkillBarsProps> = ({ accuracy, consistency, speed, durationSeconds }) => {
  // Normalize accuracy to 0-100 if it's between 0-1
  const accuracyPercent = accuracy > 1 ? accuracy : accuracy * 100;
  
  // Calculate percentages for bars
  const accuracyPercentage = Math.min(100, Math.max(0, accuracyPercent));
  const consistencyPercentage = Math.min(100, Math.max(0, consistency));
  const speedPercentage = Math.min(100, Math.max(0, (speed / 200) * 100)); // Max 200 WPM for visualization

  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800 h-full flex flex-col justify-between">
      {/* Large Numbers Section */}
      <div className="mb-0">
        {/* Accuracy */}
        <div className="flex items-baseline gap-3 mb-4">
          <p className="text-9xl font-bold text-emerald-500 font-mono">{Math.round(accuracyPercent)}<span className="text-3xl">%</span></p>
          <p className="text-sm text-gray-500 font-mono">Accuracy</p>
        </div>
        
        {/* Consistency */}
        <div className="flex items-baseline gap-3 mb-4">
          <p className="text-9xl font-bold text-blue-500 font-mono">{Math.round(consistency)}</p>
          <p className="text-sm text-gray-500 font-mono">Consistency</p>
        </div>
        
        {/* Speed */}
        <div className="flex items-baseline gap-3 mb-4">
          <p className="text-9xl font-bold text-purple-500 font-mono">{Math.round(speed)}</p>
          <p className="text-sm text-gray-500 font-mono">WPM</p>
        </div>

        {durationSeconds !== undefined && (
          <div className="flex items-baseline gap-3">
            <p className="text-9xl font-bold text-white font-mono">{Math.round(durationSeconds)}</p>
            <p className="text-xs text-gray-300 font-mono">sec</p>
          </div>
        )}
      </div>

      {/* Bars Section (No Labels) */}
      <div className="space-y-4">
        {/* Accuracy Bar */}
        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
          <div 
            className="h-full bg-emerald-500 transition-all duration-1000 ease-out" 
            style={{ width: `${accuracyPercentage}%` }} 
          />
        </div>
        
        {/* Consistency Bar */}
        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 transition-all duration-1000 ease-out" 
            style={{ width: `${consistencyPercentage}%` }} 
          />
        </div>
        
        {/* Speed Bar */}
        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
          <div 
            className="h-full bg-purple-500 transition-all duration-1000 ease-out" 
            style={{ width: `${speedPercentage}%` }} 
          />
        </div>
      </div>
    </div>
  );
};

export default SkillBars;