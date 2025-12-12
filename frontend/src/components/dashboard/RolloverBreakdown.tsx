import React from 'react';

interface RolloverBreakdownProps {
  rolloverL2L: number;
  rolloverR2R: number;
  rolloverCross: number;
  rollover: number;
}

const RolloverBreakdown: React.FC<RolloverBreakdownProps> = ({
  rolloverL2L,
  rolloverR2R,
  rolloverCross,
  rollover
}) => {
  return (
    <div className="bg-gray-900 rounded-xl p-6 min-h-[300px] border border-gray-800">
      <h3 className="text-gray-400 text-sm font-medium mb-6">Rollover Breakdown</h3>
      <div className="flex justify-around items-center gap-4 min-h-[180px]">
        {/* L to L */}
        <div className="flex flex-col items-center justify-center h-full">
          <svg width="160" height="160" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="50" fill="none" stroke="#374151" strokeWidth="12" />
            <circle 
              cx="60" 
              cy="60" 
              r="50" 
              fill="none" 
              stroke="#3B82F6" 
              strokeWidth="12"
              strokeDasharray={`${(rolloverL2L / 100) * 314.159} 314.159`}
              strokeLinecap="round"
              transform="rotate(-90 60 60)"
            />
            <text x="60" y="52" textAnchor="middle" dominantBaseline="central" fontSize="24" fontWeight="bold" fill="white">{rolloverL2L.toFixed(0)}%</text>
            <text x="60" y="74" textAnchor="middle" dominantBaseline="central" fontSize="14" fill="#9CA3AF">L → L</text>
          </svg>
        </div>

        {/* R to R */}
        <div className="flex flex-col items-center justify-center h-full">
          <svg width="160" height="160" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="50" fill="none" stroke="#374151" strokeWidth="12" />
            <circle 
              cx="60" 
              cy="60" 
              r="50" 
              fill="none" 
              stroke="#10B981" 
              strokeWidth="12"
              strokeDasharray={`${(rolloverR2R / 100) * 314.159} 314.159`}
              strokeLinecap="round"
              transform="rotate(-90 60 60)"
            />
            <text x="60" y="52" textAnchor="middle" dominantBaseline="central" fontSize="24" fontWeight="bold" fill="white">{rolloverR2R.toFixed(0)}%</text>
            <text x="60" y="74" textAnchor="middle" dominantBaseline="central" fontSize="14" fill="#9CA3AF">R → R</text>
          </svg>
        </div>

        {/* Cross */}
        <div className="flex flex-col items-center justify-center h-full">
          <svg width="160" height="160" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="50" fill="none" stroke="#374151" strokeWidth="12" />
            <circle 
              cx="60" 
              cy="60" 
              r="50" 
              fill="none" 
              stroke="#F59E0B" 
              strokeWidth="12"
              strokeDasharray={`${(rolloverCross / 100) * 314.159} 314.159`}
              strokeLinecap="round"
              transform="rotate(-90 60 60)"
            />
            <text x="60" y="52" textAnchor="middle" dominantBaseline="central" fontSize="24" fontWeight="bold" fill="white">{rolloverCross.toFixed(0)}%</text>
            <text x="60" y="74" textAnchor="middle" dominantBaseline="central" fontSize="14" fill="#9CA3AF">Cross</text>
          </svg>
        </div>

        {/* Overall */}
        <div className="flex flex-col items-center justify-center h-full">
          <svg width="160" height="160" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="50" fill="none" stroke="#374151" strokeWidth="12" />
            <circle 
              cx="60" 
              cy="60" 
              r="50" 
              fill="none" 
              stroke="#A855F7" 
              strokeWidth="12"
              strokeDasharray={`${(rollover / 100) * 314.159} 314.159`}
              strokeLinecap="round"
              transform="rotate(-90 60 60)"
            />
            <text x="60" y="52" textAnchor="middle" dominantBaseline="central" fontSize="24" fontWeight="bold" fill="white">{rollover.toFixed(0)}%</text>
            <text x="60" y="74" textAnchor="middle" dominantBaseline="central" fontSize="14" fill="#9CA3AF">Overall</text>
          </svg>
        </div>
      </div>
    </div>
  );
};

export default RolloverBreakdown;
