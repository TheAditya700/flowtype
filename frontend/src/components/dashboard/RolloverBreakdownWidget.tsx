import React from 'react';

interface RolloverBreakdownWidgetProps {
  overall: number;
  l2l: number;
  r2r: number;
  cross: number;
}

const RolloverBreakdownWidget: React.FC<RolloverBreakdownWidgetProps> = ({ overall, l2l, r2r, cross }) => {
  const getColor = (value: number) => {
    if (value < 10) return 'text-green-400';
    if (value < 20) return 'text-yellow-400';
    return 'text-red-400';
  };

  const RolloverItem: React.FC<{ label: string; value: number }> = ({ label, value }) => (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1">
        <span className="text-gray-400 text-sm">{label}</span>
        <span className={`font-mono font-bold ${getColor(value)}`}>{Math.round(value)}%</span>
      </div>
      <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all duration-1000 ease-out ${getColor(value).replace('text-', 'bg-')}`}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800 h-full flex flex-col justify-center">
      <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-6">Rollover Breakdown</h3>
      
      {/* Overall */}
      <div className="mb-6 pb-6 border-b border-gray-700">
        <RolloverItem label="Overall" value={overall} />
      </div>

      {/* Per-Hand */}
      <RolloverItem label="Left Hand (L2L)" value={l2l} />
      <RolloverItem label="Right Hand (R2R)" value={r2r} />
      <RolloverItem label="Cross Hand" value={cross} />
    </div>
  );
};

export default RolloverBreakdownWidget;
