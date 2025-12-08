import React from 'react';
import { UserProfile } from '../../types';

interface ProgressChartsProps {
  userProfile: UserProfile | null;
}

const ProgressCharts: React.FC<ProgressChartsProps> = ({ userProfile }) => {
  if (!userProfile) {
    return (
      <div className="bg-container p-6 rounded-xl shadow-lg flex flex-col justify-center items-center h-full">
        <p className="text-subtle text-lg">Login to track your progress over time!</p>
      </div>
    );
  }

  // Placeholder for actual charts
  return (
    <div className="bg-container p-6 rounded-xl shadow-lg flex flex-col">
      <h3 className="text-xl font-semibold text-text mb-4">Progress Trends</h3>
      <div className="flex-grow flex items-center justify-center text-subtle text-center h-48">
        <p>Historical charts for WPM, Accuracy, and Flow will appear here.</p>
      </div>
      {/* Example: A simple list of stats from userProfile (if available) */}
      {userProfile.stats && (
        <div className="mt-4 text-sm text-text-light">
          <p>Overall Average WPM: {userProfile.stats.avg_wpm ? userProfile.stats.avg_wpm.toFixed(1) : 'N/A'}</p>
          <p>Overall Average Accuracy: {userProfile.stats.avg_accuracy ? (userProfile.stats.avg_accuracy * 100).toFixed(1) : 'N/A'}%</p>
          <p>Total Sessions: {userProfile.stats.total_sessions}</p>
        </div>
      )}
    </div>
  );
};

export default ProgressCharts;
