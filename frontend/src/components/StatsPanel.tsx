import React from 'react';

interface StatsPanelProps {
  wpm: number;
  accuracy: number; // As a percentage
}

const StatsPanel: React.FC<StatsPanelProps> = ({ wpm, accuracy }) => {
  return (
    <div className="stats-panel p-4 bg-gray-700 rounded-lg text-center">
      <h2 className="text-2xl font-semibold mb-4 text-blue-300">Current Stats</h2>
      <div className="mb-4">
        <p className="text-gray-400 text-lg">WPM</p>
        <p className="text-5xl font-bold text-white">{Math.round(wpm)}</p>
      </div>
      <div>
        <p className="text-gray-400 text-lg">Accuracy</p>
        <p className="text-5xl font-bold text-white">{Math.round(accuracy)}%</p>
      </div>
    </div>
  );
};

export default StatsPanel;
