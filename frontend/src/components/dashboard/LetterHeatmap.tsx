import React, { useMemo } from 'react';
import { KeystrokeEvent } from '../../types'; // Corrected path

interface LetterHeatmapProps { // Renamed interface
  keystrokes: KeystrokeEvent[];
}

const KEYS_ROW1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'];
const KEYS_ROW2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"];
const KEYS_ROW3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'];

const LetterHeatmap: React.FC<LetterHeatmapProps> = ({ keystrokes }) => { // Renamed component
  // 1. Aggregate data per key
  const keyStats = useMemo(() => {
    const stats: Record<string, { total: number; errors: number; latencySum: number }> = {};

    keystrokes.forEach(k => {
      const char = k.key.toLowerCase();
      if (!stats[char]) {
        stats[char] = { total: 0, errors: 0, latencySum: 0 };
      }
      stats[char].total += 1;
      if (!k.isCorrect) {
        stats[char].errors += 1;
      }
    });
    return stats;
  }, [keystrokes]);

  // 2. Color scale logic
  const getKeyColor = (key: string) => {
    const s = keyStats[key];
    if (!s || s.total === 0) return 'bg-gray-700 text-gray-400'; // Untouched

    const accuracy = 1 - (s.errors / s.total);
    
    // Color Gradient: Red (0) -> Yellow (0.8) -> Green (1.0)
    if (accuracy === 1) return 'bg-green-500 text-white';
    if (accuracy >= 0.95) return 'bg-green-400 text-black';
    if (accuracy >= 0.9) return 'bg-yellow-400 text-black';
    if (accuracy >= 0.8) return 'bg-yellow-500 text-black';
    return 'bg-red-500 text-white';
  };

  const renderKey = (key: string, widthClass: string = 'w-14') => {
    const s = keyStats[key];
    const accuracy = s ? ((1 - s.errors / s.total) * 100).toFixed(0) : 0;
    
    return (
      <div 
        key={key}
        className={`${widthClass} h-14 flex items-center justify-center rounded m-1 text-lg font-bold transition-colors ${getKeyColor(key)}`}
        title={s ? `Key: ${key.toUpperCase()}\nAccuracy: ${accuracy}%\nTyped: ${s.total}` : 'No data'}
      >
        {key.toUpperCase()}
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center p-4 bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-gray-300 mb-4 font-semibold">Confidence Heatmap</h3>
      
      <div className="flex">
        {KEYS_ROW1.map(k => renderKey(k))}
      </div>
      <div className="flex ml-4">
        {KEYS_ROW2.map(k => renderKey(k))}
      </div>
      <div className="flex ml-10">
        {KEYS_ROW3.map(k => renderKey(k))}
      </div>
      
      {/* Legend */}
      <div className="flex gap-4 mt-4 text-xs text-gray-400">
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-green-500 rounded"></div> 100%</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-yellow-400 rounded"></div> 90-99%</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-red-500 rounded"></div> &lt;90%</div>
      </div>
    </div>
  );
};

export default LetterHeatmap;
