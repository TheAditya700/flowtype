import React from 'react';

const ProgressChart: React.FC = () => {
  return (
    <div className="progress-chart p-4 bg-gray-700 rounded-lg text-center">
      <h2 className="text-2xl font-semibold mb-4 text-blue-300">Difficulty Progression</h2>
      <p className="text-gray-400">Chart showing difficulty over time.</p>
    </div>
  );
};

export default ProgressChart;
