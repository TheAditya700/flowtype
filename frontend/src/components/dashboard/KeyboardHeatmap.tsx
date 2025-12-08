import React, { useState } from 'react';

interface KeyStats {
  char: string;
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
    if (!stats) return 'bg-gray-800 text-gray-600';

    const val = mode === 'accuracy' ? stats.accuracy : stats.speed;
    
    // Gradient logic (simplified)
    // Accuracy: Red (low) -> Green (high)
    // Speed: Blue (slow) -> Orange/White (fast)
    
    if (mode === 'accuracy') {
        if (val >= 0.95) return 'bg-green-500/20 text-green-400 border-green-500/50';
        if (val >= 0.90) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
        return 'bg-red-500/20 text-red-400 border-red-500/50';
    } else {
        if (val >= 0.8) return 'bg-blue-400/20 text-blue-300 border-blue-400/50';
        if (val >= 0.5) return 'bg-blue-600/20 text-blue-500 border-blue-600/50';
        return 'bg-gray-700/50 text-gray-500 border-gray-700';
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
                  w-10 h-10 flex items-center justify-center rounded-lg border 
                  uppercase font-mono font-bold text-sm transition-all
                  ${getColor(char)}
                `}
                title={charStats[char] ? `${mode}: ${(charStats[char] as any)[mode].toFixed(2)}` : 'No Data'}
              >
                {char}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default KeyboardHeatmap;
