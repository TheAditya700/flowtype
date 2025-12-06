import React from 'react';

interface StatsWidgetProps {
  wpm: string | number;
  accuracy: string | number;
  totalWords: number;
}

const StatsWidget: React.FC<StatsWidgetProps> = ({ wpm, accuracy, totalWords }) => {
  return (
    <div className="flex flex-col gap-8 h-full justify-center">
      <div className="text-right">
        <div className="text-subtle text-xs uppercase tracking-widest">Avg WPM</div>
        <div className="text-6xl font-bold text-primary">{wpm}</div>
      </div>
      <div className="text-right">
        <div className="text-subtle text-xs uppercase tracking-widest">Acc</div>
        <div className="text-6xl font-bold text-success">{accuracy}%</div>
      </div>
      <div className="text-right">
        <div className="text-subtle text-xs uppercase tracking-widest">Words</div>
        <div className="text-4xl font-bold text-text">{totalWords}</div>
      </div>
    </div>
  );
};

export default StatsWidget;
