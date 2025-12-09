import React from 'react';

interface TypingZoneStatsDisplayProps {
  wpm: number;
  accuracy: number;
  time: number;
  sessionMode: '15' | '30' | '60' | '120' | 'free';
  timeRemaining: number | null;
}

const TypingZoneStatsDisplay: React.FC<TypingZoneStatsDisplayProps> = ({ wpm, accuracy, time, sessionMode, timeRemaining }) => {
  const formatTime = (seconds: number) => {
    const s = Math.round(seconds);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const remS = s % 60;
    return `${m}m ${remS}s`;
  };

  const getDisplayTime = () => {
    if (sessionMode !== 'free') {
      // Show countdown for timed modes
      if (timeRemaining !== null) {
        const remainingSeconds = Math.ceil(timeRemaining / 1000);
        return `${remainingSeconds}s`;
      } else {
        // Before session starts, show the total duration
        return `${sessionMode}s`;
      }
    }
    // Show elapsed time for free mode
    return formatTime(time);
  };

  const getTimeLabel = () => {
    return sessionMode !== 'free' ? 'remaining' : 'time';
  };

  return (
    <div className="flex gap-8 text-xl text-subtle">
      <div>
        <span className="text-sm text-subtle mr-2">{getTimeLabel()}</span>
        <span className="font-bold text-primary">{getDisplayTime()}</span>
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
