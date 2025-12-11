import React, { useState } from 'react';

interface KeyStats {
  accuracy: number; // 0-1
  speed: number;    // normalized 0-1 (1 is fast)
}

interface KeyboardHeatmapProps {
  charStats: Record<string, KeyStats>;
}

const ROWS = [
  ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
  ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"],
  ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
];

const KeyboardHeatmap: React.FC<KeyboardHeatmapProps> = ({ charStats }) => {
  const [mode, setMode] = useState<'accuracy' | 'speed'>('accuracy');

  const getColor = (char: string) => {
    const stats = charStats[char];
    if (!stats) return 'bg-gray-800 text-gray-700 border-gray-800';

    const val = mode === 'accuracy' ? stats.accuracy : stats.speed;
    
    // Accuracy: Red (low) -> Yellow (mid) -> Green (high)
    // Speed: Dark blue (slow) -> Mid blue -> Light blue (fast)
    if (mode === 'accuracy') {
      if (val >= 0.95) return 'bg-green-500/20 text-green-400 border-green-400';
      if (val >= 0.90) return 'bg-yellow-500/20 text-yellow-400 border-yellow-400';
      return 'bg-red-500/20 text-red-400 border-red-400';
    } else {
      if (val >= 0.8) return 'bg-blue-400/20 text-blue-300 border-blue-300';
      if (val >= 0.5) return 'bg-blue-600/20 text-blue-500 border-blue-500';
      return 'bg-blue-800/20 text-blue-700 border-blue-700';
    }
  };

  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800 flex flex-col items-center">
      <div className="flex justify-between w-full mb-6">
        <h3 className="text-gray-400 text-sm font-medium">Keyboard Heatmap</h3>
        <div className="flex bg-gray-800 rounded-lg p-1">
          <button 
            onClick={() => setMode('accuracy')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${mode === 'accuracy' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            Accuracy
          </button>
          <button 
            onClick={() => setMode('speed')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${mode === 'speed' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            Speed
          </button>
        </div>
      </div>
      
      <div className="flex h-auto flex-col p-8 pb-10 gap-2 select-none">
        {ROWS.map((row, rIdx) => (
          <div key={rIdx} className="flex justify-center gap-2">
            {row.map((char) => (
              <div
                key={char}
                className={`
                  w-16 h-16 flex items-center justify-center rounded-lg border 
                  uppercase font-mono text-xl font-bold text-sm transition-all
                  ${getColor(char)}
                `}
                title={
                  charStats[char]
                    ? `${mode}: ${(charStats[char] as any)[mode].toFixed(2)}`
                    : 'No data'
                }
              >
                {char}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Legend (3 boxes with explicit ranges, left-aligned) */}
      <div className="mt-2 w-full flex justify-start">
        {mode === 'accuracy' ? (
          <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <div className="w-4 h-3 bg-red-500 rounded-sm" />
            <span>{"< 90%"}</span>
          </div>

          <div className="flex items-center gap-1">
            <div className="w-4 h-3 bg-yellow-500 rounded-sm" />
            <span>{"90–94%"}</span>
          </div>

          <div className="flex items-center gap-1">
            <div className="w-4 h-3 bg-green-500 rounded-sm" />
            <span>{">= 95%"}</span>
          </div>
        </div>

        ) : (
         <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <div className="w-4 h-3 bg-blue-800 rounded-sm" />
            <span>{"Slow (< 0.50)"}</span>
          </div>

          <div className="flex items-center gap-1">
            <div className="w-4 h-3 bg-blue-600 rounded-sm" />
            <span>{"Medium (0.50–0.79)"}</span>
          </div>

          <div className="flex items-center gap-1">
            <div className="w-4 h-3 bg-blue-400 rounded-sm" />
            <span>{"Fast (>= 0.80)"}</span>
          </div>
        </div>

        )}
      </div>
    </div>
  );
};

export default KeyboardHeatmap;
