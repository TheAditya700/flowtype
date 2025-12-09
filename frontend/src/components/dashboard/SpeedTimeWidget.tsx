import React from 'react';

interface SpeedTimeWidgetProps {
  wpm: number;
  rawWpm: number;
  durationSeconds: number;
}

const SpeedTimeWidget: React.FC<SpeedTimeWidgetProps> = ({ wpm, rawWpm, durationSeconds }) => {
  const minutes = Math.floor(durationSeconds / 60);
  const seconds = Math.floor(durationSeconds % 60);

  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800 h-full flex flex-col justify-between">
      {/* Main WPM (Very Big) */}
      <div className="mb-2">
        <div className="flex items-baseline gap-2 mb-2">
          <p className="text-9xl font-bold text-blue-400 font-mono">{Math.round(wpm)}</p>
          <p className="text-xl text-gray-500 font-mono">WPM</p>
        </div>
      </div>

      {/* Raw WPM (Same size as WPM label) */}
      <div className="mb-8 flex justify-between items-end">
        <div className="flex items-baseline gap-2">
          <p className="text-7xl text-gray-400 font-mono">{Math.round(rawWpm)}</p>
          <p className="text-sm text-gray-500 font-mono">Raw</p>
        </div>
        
        {/* Time (Inline with Raw WPM, bottom right) */}
        <div className="flex flex-col items-end">
          <p className="text-3xl font-bold text-gray-200 font-mono">
            {minutes}:{seconds.toString().padStart(2, '0')}
          </p>
          <p className="text-xs text-gray-500 font-mono">mm:ss</p>
        </div>
      </div>
    </div>
  );
};

export default SpeedTimeWidget;
