import React from 'react';

interface SessionStatsWidgetProps {
  errors: number;
  rawWpm: number;
  kspc: number;
  avgChunkLength: number;
  rollover: number;
  smoothness: number;
}

const SessionStatsWidget: React.FC<SessionStatsWidgetProps> = ({
  errors,
  rawWpm,
  kspc,
  avgChunkLength,
  rollover
}) => {
  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800">
      <div className="grid grid-cols-5 gap-6">

        {/* Errors */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-4xl font-bold font-mono text-red-400">
            {errors}
          </div>
          <div className="text-xs text-gray-500 font-mono mt-1">Errors</div>
        </div>

        {/* Raw WPM */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-4xl font-bold font-mono text-purple-400">
            {Math.round(rawWpm)}
          </div>
          <div className="text-xs text-gray-500 font-mono mt-1">Raw WPM</div>
        </div>

        {/* KSPC */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-4xl font-bold font-mono text-white-400">
            {kspc.toFixed(2)}
          </div>
          <div className="text-xs text-gray-500 font-mono mt-1">KSPC</div>
        </div>

        {/* Avg Chunk Length */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-4xl font-bold font-mono text-emerald-400">
            {avgChunkLength.toFixed(1)}
          </div>
          <div className="text-xs text-gray-500 font-mono mt-1">Chunk Len</div>
        </div>

        {/* Rollover */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-4xl font-bold font-mono text-yellow-400">
            {rollover.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500 font-mono mt-1">Rollover</div>
        </div>

      </div>
    </div>
  );
};

export default SessionStatsWidget;
