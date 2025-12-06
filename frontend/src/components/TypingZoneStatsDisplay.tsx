import React from 'react';

interface TypingZoneStatsDisplayProps {
  wpm: number;
  accuracy: number;
}

const TypingZoneStatsDisplay: React.FC<TypingZoneStatsDisplayProps> = ({ wpm, accuracy }) => {
  return (
    <div className="flex gap-8 text-xl text-subtle">
      <div>
        <span className="text-sm text-subtle mr-2">wpm</span>
        <span className="font-bold text-primary">{Math.round(wpm)}</span>
      </div>
      <div>
        <span className="text-sm text-subtle mr-2">acc</span>
        <span className="font-bold text-success">{Math.round(accuracy)}%</span>
      </div>
    </div>
  );
};

export default TypingZoneStatsDisplay;
