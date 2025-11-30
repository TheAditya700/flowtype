import React from 'react';
import { TypingSession } from '../types';

interface SessionHistoryProps {
  sessions: TypingSession[];
}

const SessionHistory: React.FC<SessionHistoryProps> = ({ sessions }) => {
  return (
    <div className="session-history p-4 bg-gray-700 rounded-lg">
      <h2 className="text-2xl font-semibold mb-4 text-blue-300">Session History</h2>
      {sessions.length === 0 ? (
        <p className="text-gray-400">No sessions yet.</p>
      ) : (
        <ul>
          {sessions.map((session, index) => (
            <li key={index} className="mb-2 p-2 border-b border-gray-600 last:border-b-0">
              <p className="text-lg text-white">WPM: {Math.round(session.wpm)}</p>
              <p className="text-gray-400">Accuracy: {Math.round(session.accuracy * 100)}%</p>
              <p className="text-gray-500 text-sm">{new Date(session.startedAt).toLocaleString()}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SessionHistory;
