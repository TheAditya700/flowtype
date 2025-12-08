import React from 'react';
import { MoveLeft, MoveRight, Shuffle } from 'lucide-react';

interface ConfidenceWidgetProps {
  left: number;  // 0-100
  right: number; // 0-100
  cross: number; // 0-100
}

const ConfidenceItem: React.FC<{ label: string; value: number; icon: React.ReactNode }> = ({ label, value, icon }) => {
  // Color based on value
  const getColor = (v: number) => {
    if (v >= 80) return 'text-emerald-400';
    if (v >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex items-center justify-between bg-gray-800/30 p-3 rounded-lg border border-gray-700/50">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-md bg-gray-800 text-gray-400`}>
          {icon}
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider">{label}</span>
          <div className="h-1.5 w-24 bg-gray-800 rounded-full mt-1 overflow-hidden">
             <div className={`h-full rounded-full ${getColor(value).replace('text-', 'bg-')} transition-all duration-1000`} style={{ width: `${value}%` }} />
          </div>
        </div>
      </div>
      <span className={`text-lg font-bold ${getColor(value)}`}>{Math.round(value)}</span>
    </div>
  );
};

const ConfidenceWidget: React.FC<ConfidenceWidgetProps> = ({ left, right, cross }) => {
  return (
    <div className="w-full bg-gray-900 rounded-xl p-4 border border-gray-800 flex flex-col gap-3 h-full justify-center">
      <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Hand Confidence</h3>
      <ConfidenceItem label="Left Hand" value={left} icon={<MoveLeft size={16} />} />
      <ConfidenceItem label="Right Hand" value={right} icon={<MoveRight size={16} />} />
      <ConfidenceItem label="Cross Hand" value={cross} icon={<Shuffle size={16} />} />
    </div>
  );
};

export default ConfidenceWidget;
