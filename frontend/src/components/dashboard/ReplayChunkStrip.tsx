import React, { useState, useEffect, useMemo } from 'react';
import { ReplayEvent } from '../../types';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface ReplayChunkStripProps {
  events: ReplayEvent[];
  text?: string; 
}

const SNIPPETS_PER_PAGE = 2;

const ReplayChunkStrip: React.FC<ReplayChunkStripProps> = ({ events }) => {
  const [page, setPage] = useState(0);
  const [mode, setMode] = useState<'replay' | 'rollover'>('replay');
  
  useEffect(() => {
    setPage(0);
  }, [events]);

  // Group events by snippetIndex (default 0 if null)
  const groupedEvents = useMemo(() => {
      const groups: Record<number, ReplayEvent[]> = {};
      let maxIdx = 0;
      events.forEach(e => {
          const idx = e.snippetIndex ?? 0;
          if (!groups[idx]) groups[idx] = [];
          groups[idx].push(e);
          if (idx > maxIdx) maxIdx = idx;
      });
      return { groups, maxIdx };
  }, [events]);

  const totalSnippets = groupedEvents.maxIdx + 1;
  const totalPages = Math.ceil(totalSnippets / SNIPPETS_PER_PAGE);
  
  const startSnippetIdx = page * SNIPPETS_PER_PAGE;
  const endSnippetIdx = startSnippetIdx + SNIPPETS_PER_PAGE;
  
  const visibleEvents = events.filter(e => {
      const idx = e.snippetIndex ?? 0;
      return idx >= startSnippetIdx && idx < endSnippetIdx;
  });

  const canGoPrev = page > 0;
  const canGoNext = page < totalPages - 1;

  const getBarColor = (m: ReplayEvent) => {
      if (mode === 'rollover') {
          return m.isRollover ? 'bg-purple-500' : 'bg-gray-700';
      }
      // Replay Mode
      if (m.iki > 300) return 'bg-red-500';
      if (m.iki > 150) return 'bg-yellow-500';
      return 'bg-green-500';
  };

  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800">
      <div className="flex justify-between items-center mb-6">
        <div className="flex bg-gray-800 rounded-lg p-1">
            <button 
              onClick={() => setMode('replay')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${mode === 'replay' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
            >
                Chunking
            </button>
            <button 
              onClick={() => setMode('rollover')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${mode === 'rollover' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
            >
                Rollover
            </button>
        </div>
        
        {/* Pagination Controls */}
        <div className="flex items-center gap-4">
            <span className="text-xs text-gray-500 font-mono">
                Snippets {startSnippetIdx + 1}-{Math.min(endSnippetIdx, totalSnippets)} / {totalSnippets}
            </span>
            <div className="flex gap-1">
                <button 
                    onClick={() => setPage(p => Math.max(0, p - 1))}
                    disabled={!canGoPrev}
                    className={`p-1.5 rounded transition-colors ${!canGoPrev ? 'text-gray-700 cursor-not-allowed' : 'text-gray-300 hover:bg-gray-800'}`}
                >
                    <ChevronLeft size={16} />
                </button>
                <button 
                    onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                    disabled={!canGoNext}
                    className={`p-1.5 rounded transition-colors ${!canGoNext ? 'text-gray-700 cursor-not-allowed' : 'text-gray-300 hover:bg-gray-800'}`}
                >
                    <ChevronRight size={16} />
                </button>
            </div>
        </div>
      </div>
      
      <div className="flex flex-wrap items-end content-start gap-y-6 min-h-[100px]">
        {visibleEvents.map((m, i) => {
          const heightPx = Math.min(50, (m.iki / 500) * 50);
          const color = getBarColor(m);

          const prevSnippetIndex = i > 0 ? visibleEvents[i-1].snippetIndex : m.snippetIndex;
          const isNewSnippet = m.snippetIndex !== prevSnippetIndex && i !== 0;

          return (
            <React.Fragment key={`char-fragment-${i}`}>
              {isNewSnippet && (
                <div key={`snippet-break-${i}`} className="w-full h-px bg-gray-700 my-4" />
              )}
              <div 
                  key={`char-event-${i}`} 
                  className={`flex flex-col items-center justify-end group relative ${m.isChunkStart && mode === 'replay' ? 'ml-3' : ''}`}
                  title={`${Math.round(m.iki)}ms - ${m.char} ${m.isRollover ? '(Rollover)' : ''}`}
              >
                  {/* Bar */}
                  <div 
                      className={`w-2 ${color} opacity-80 rounded-t-sm mb-1 transition-all`}
                      style={{ height: `${Math.max(4, heightPx)}px` }}
                  />
                  
                  {/* Character */}
                  <span 
                      className={`
                        font-mono text-xl leading-none px-[2px] border-b-2 min-w-[1ch] text-center
                        ${m.isError ? 'text-red-400 border-red-500' : 'text-gray-200 border-transparent'}
                        ${m.isChunkStart && mode === 'replay' ? 'border-l-2 border-l-blue-500/50' : ''}
                      `}
                  >
                      {m.char === ' ' ? '\u00A0' : m.char}
                  </span>
              </div>
            </React.Fragment>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="mt-4 flex gap-4 text-xs text-gray-500">
          {mode === 'rollover' ? (
              <>
                <div className="flex items-center gap-1"><div className="w-3 h-3 bg-purple-500 rounded-sm"></div> Rollover</div>
                <div className="flex items-center gap-1"><div className="w-3 h-3 bg-gray-700 rounded-sm"></div> No Rollover</div>
              </>
          ) : (
             <>
                <div className="flex items-center gap-1"><div className="w-3 h-3 bg-green-500 rounded-sm"></div> &lt; 150ms</div>
                <div className="flex items-center gap-1"><div className="w-3 h-3 bg-yellow-500 rounded-sm"></div> 150-300ms</div>
                <div className="flex items-center gap-1"><div className="w-3 h-3 bg-red-500 rounded-sm"></div> &gt; 300ms</div>
             </>
          )}
      </div>
    </div>
  );
};

export default ReplayChunkStrip;
