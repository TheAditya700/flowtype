import React from 'react';

interface TypingZoneStatsDisplayProps {
  wpm: number;
  accuracy: number;
  time: number;
}

const TypingZoneStatsDisplay: React.FC<TypingZoneStatsDisplayProps> = ({ wpm, accuracy, time }) => {
  const formatTime = (seconds: number) => {
    const s = Math.round(seconds);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const remS = s % 60;
    return `${m}m ${remS}s`;
  };

  return (
    <div className="flex gap-8 text-xl text-subtle">
      <div>
        <span className="text-sm text-subtle mr-2">time</span>
        <span className="font-bold text-primary">{formatTime(time)}</span>
      </div>
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
