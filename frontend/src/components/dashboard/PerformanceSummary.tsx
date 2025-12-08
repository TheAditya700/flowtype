import React from 'react';
import { UserProfile } from '../../types';

interface PerformanceSummaryProps {
  wpm: number;
  accuracy: number;
  duration: number; // in seconds
  userProfile: UserProfile | null; // For skill comparison
}

const PerformanceSummary: React.FC<PerformanceSummaryProps> = ({ wpm, accuracy, duration, userProfile }) => {
  // Simple emoji logic for now
  const getPerformanceEmoji = (acc: number) => {
    if (acc > 98) return '🔥 Flowing';
    if (acc > 90) return '🙂 Smooth';
    if (acc > 70) return '😐 Rough patch';
    return '😵 Struggling';
  };

  const formattedAccuracy = (accuracy * 100).toFixed(1); // Assuming accuracy is 0-1
  const formattedDuration = `${Math.floor(duration / 60)}m ${Math.round(duration % 60)}s`;

  return (
    <div className="bg-container p-6 rounded-xl shadow-lg flex flex-col items-center justify-center text-center">
      <h2 className="text-6xl font-extrabold text-primary mb-4">{wpm} WPM</h2>
      <p className="text-3xl text-text-light mb-4">Accuracy: {formattedAccuracy}%</p>
      <p className="text-xl text-subtle mb-6">Time: {formattedDuration}</p>
      
      <div className="text-4xl mb-4">
        {getPerformanceEmoji(accuracy * 100)}
      </div>

      {userProfile && (
        <div className="text-subtle text-md">
          Difficulty vs. Skill: <span className="text-primary">Comfort Zone</span>
        </div>
      )}
    </div>
  );
};

export default PerformanceSummary;
