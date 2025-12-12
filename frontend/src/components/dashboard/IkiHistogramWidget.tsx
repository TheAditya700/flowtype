import React, { useState, useMemo } from 'react';
import { ReplayEvent } from '../../types';

interface IkiHistogramWidgetProps {
  replayEvents: ReplayEvent[];
}

const IkiHistogramWidget: React.FC<IkiHistogramWidgetProps> = ({ replayEvents }) => {
  const [filter, setFilter] = useState<'all' | 'left' | 'right'>('all');

  const leftHandKeys = new Set('12345qwertasdfgzxcvb'.split(''));
  const rightHandKeys = new Set('67890yuiophjklnm'.split(''));

  const getHand = (char: string): 'L' | 'R' | null => {
    const lower = char.toLowerCase();
    if (leftHandKeys.has(lower)) return 'L';
    if (rightHandKeys.has(lower)) return 'R';
    return null;
  };

  const filteredEvents = useMemo(() => {
    if (filter === 'all') return replayEvents.filter(e => e.iki > 0);
    if (filter === 'left') return replayEvents.filter(e => getHand(e.char) === 'L' && e.iki > 0);
    if (filter === 'right') return replayEvents.filter(e => getHand(e.char) === 'R' && e.iki > 0);
    return replayEvents.filter(e => e.iki > 0);
  }, [replayEvents, filter]);

  // Build histogram: 50ms bins from 0-500ms (capped)
  const BIN_SIZE = 50;
  const MAX_IKI = 500;
  const binCount = Math.ceil(MAX_IKI / BIN_SIZE);
  const bins = useMemo(() => {
    const b = Array(binCount).fill(0);
    filteredEvents.forEach(e => {
      const cappedIki = Math.min(e.iki, MAX_IKI - 1);
      const binIdx = Math.floor(cappedIki / BIN_SIZE);
      if (binIdx >= 0 && binIdx < binCount) b[binIdx]++;
    });
    return b;
  }, [filteredEvents, binCount]);

  const maxCount = Math.max(...bins, 1);

  {/* text-gray-400 text-sm font-medium absolute top-6 left-6 */}

  return (
    <div className="w-full h-full bg-gray-900 rounded-xl p-6 border border-gray-800 flex flex-col gap-4">
      {/* Header with title and filter tabs inline, right-aligned */}
      <div className="flex w-full justify-between">
        <h3 className="text-gray-400 text-sm font-medium">IKI Distribution</h3>
        <div className="flex bg-gray-800 rounded-lg p-1 w-fit">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              filter === 'all' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilter('left')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              filter === 'left' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            Left
          </button>
          <button
            onClick={() => setFilter('right')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              filter === 'right' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            Right
          </button>
        </div>
      </div>

      {/* Histogram bars with labels - stretched height */}
      <div className="flex-1 flex items-end gap-1 min-h-[150px]">
        {bins.map((count, idx) => {
          const heightPercent = (count / maxCount) * 100;
          const binStart = idx * BIN_SIZE;
          const binEnd = binStart + BIN_SIZE;
          return (
            <div key={`bin-${idx}`} className="flex-1 flex flex-col items-center justify-end h-full gap-2 group">
              <div
                className="w-full bg-purple-500 rounded-t-sm hover:bg-purple-400 transition-colors relative"
                style={{ height: `${heightPercent}%`, minHeight: count > 0 ? '2px' : '0px' }}
                title={`${binStart}-${binEnd}ms: ${count} events`}
              />
              {/* Bucket label */}
              <div className="text-[9px] text-gray-500 whitespace-nowrap text-center">
                {binStart}-{binEnd}
              </div>
            </div>
          );
        })}
      </div>

      {/* Stats */}
      <div className="text-xs text-gray-500 flex justify-between border-t border-gray-700 pt-2">
        <span>{filteredEvents.length} events</span>
        <span>Avg: {filteredEvents.length > 0 ? Math.round(filteredEvents.reduce((s, e) => s + e.iki, 0) / filteredEvents.length) : 0}ms</span>
      </div>
    </div>
  );
};

export default IkiHistogramWidget;
