import React from 'react';

const ResultCard: React.FC = () => {
  return (
    <div className="result-card p-4 bg-gray-700 rounded-lg text-center">
      <h2 className="text-2xl font-semibold mb-4 text-blue-300">Session Result</h2>
      <p className="text-gray-400">This will be a canvas-generated image for sharing.</p>
    </div>
  );
};

export default ResultCard;
