import React, { useState, useMemo } from 'react';
import LetterHeatmap from './LetterHeatmap'; // Assuming LetterHeatmap.tsx is in the same directory
import { KeystrokeEvent } from '../../types';

interface WeaknessHeatmapsProps {
  keystrokeEvents: KeystrokeEvent[];
}

// Placeholder for BigramHeatmap
const BigramHeatmap: React.FC<WeaknessHeatmapsProps> = ({ keystrokeEvents }) => {
    const bigramStats = useMemo(() => {
        const stats: Record<string, { total: number; errors: number }> = {};
        if (keystrokeEvents.length < 2) return stats;

        for (let i = 1; i < keystrokeEvents.length; i++) {
            const prevKey = keystrokeEvents[i - 1].key.toLowerCase();
            const currentKey = keystrokeEvents[i].key.toLowerCase();
            if (prevKey.length !== 1 || currentKey.length !== 1 || prevKey === ' ' || currentKey === ' ') continue; // Only single chars for now

            const bigram = prevKey + currentKey;
            if (!stats[bigram]) {
                stats[bigram] = { total: 0, errors: 0 };
            }
            stats[bigram].total += 1;
            // An error on the second character of a bigram
            if (!keystrokeEvents[i].isCorrect) {
                stats[bigram].errors += 1;
            }
        }
        
        // Sort and take top N weakest
        const sortedBigrams = Object.entries(stats)
            .map(([bigram, s]) => ({
                bigram,
                accuracy: s.total > 0 ? (1 - s.errors / s.total) : 1,
                total: s.total
            }))
            .filter(b => b.total > 2) // Only consider bigrams typed more than twice
            .sort((a, b) => a.accuracy - b.accuracy) // Ascending accuracy (weakest first)
            .slice(0, 10); // Top 10 weakest

        return sortedBigrams;
    }, [keystrokeEvents]);

    return (
        <div className="p-4">
            <h4 className="text-lg font-semibold text-text mb-2">Top Weakest Bigrams</h4>
            {bigramStats.length > 0 ? (
                <ul className="space-y-2">
                    {bigramStats.map((b, index) => (
                        <li key={b.bigram} className="flex justify-between items-center bg-bg p-2 rounded-md">
                            <span className="font-mono text-xl text-primary">{b.bigram}</span>
                            <span className={`text-sm ${b.accuracy < 0.8 ? 'text-error' : 'text-success'}`}>
                                {(b.accuracy * 100).toFixed(1)}% ({b.total})
                            </span>
                        </li>
                    ))}
                </ul>
            ) : (
                <p className="text-subtle text-sm">No significant bigram data yet.</p>
            )}
        </div>
    );
};


const WeaknessHeatmaps: React.FC<WeaknessHeatmapsProps> = ({ keystrokeEvents }) => {
  const [activeTab, setActiveTab] = useState<'letters' | 'bigrams' | 'phrases'>('letters');

  return (
    <div className="bg-container p-6 rounded-xl shadow-lg flex flex-col">
      <h3 className="text-xl font-semibold text-text mb-4">Weakness Heatmaps</h3>
      
      <div className="flex border-b border-gray-700 mb-4">
        <button 
          className={`py-2 px-4 text-sm font-medium ${activeTab === 'letters' ? 'text-primary border-b-2 border-primary' : 'text-subtle hover:text-text'}`}
          onClick={() => setActiveTab('letters')}
        >
          Letters
        </button>
        <button 
          className={`py-2 px-4 text-sm font-medium ${activeTab === 'bigrams' ? 'text-primary border-b-2 border-primary' : 'text-subtle hover:text-text'}`}
          onClick={() => setActiveTab('bigrams')}
        >
          Bigrams
        </button>
        <button 
          className={`py-2 px-4 text-sm font-medium ${activeTab === 'phrases' ? 'text-primary border-b-2 border-primary' : 'text-subtle hover:text-text'}`}
          onClick={() => setActiveTab('phrases')}
          disabled // Phrase weakness map is not implemented yet
        >
          Phrases (Soon)
        </button>
      </div>

      <div className="flex-grow">
        {activeTab === 'letters' && <LetterHeatmap keystrokes={keystrokeEvents} />}
        {activeTab === 'bigrams' && <BigramHeatmap keystrokes={keystrokeEvents} />}
        {activeTab === 'phrases' && (
          <div className="text-subtle text-center py-8">Phrase analysis coming soon!</div>
        )}
      </div>
    </div>
  );
};

export default WeaknessHeatmaps;
