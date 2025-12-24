import React from "react";

interface Props {
  percent: number; // 0-100
  loading?: boolean;
}

const clamp = (v: number, min = 0, max = 100) => Math.max(min, Math.min(max, v));

const WeightsUpdatedGauge: React.FC<Props> = ({ percent, loading = false }) => {
  if (loading) {
    return (
      <div className="h-full bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Gauge...</span>
      </div>
    );
  }

  const p = clamp(percent);
  const circumference = 2 * Math.PI * 50; // r=50
  const dash = (p / 100) * circumference;

  // Color scale: green for high update rate, orange mid, blue low
  const color = "#F59E0B";

  return (
    <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col h-full">
      <div className="mb-2">
        <h3 className="text-lg font-semibold text-white">Weights Updated</h3>
        <p className="text-xs text-gray-400 font-mono">Percent of weights changed</p>
      </div>
      <div className="flex-grow h-[250px] flex items-center justify-center">
        <svg width="200" height="200" viewBox="0 0 120 120">
          <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="1.5" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <circle cx="60" cy="60" r="50" fill="none" stroke="#374151" strokeWidth="12" />
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeDasharray={`${dash} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
            filter="url(#glow)"
          />
          <text
            x="60"
            y="52"
            textAnchor="middle"
            dominantBaseline="central"
            fontSize="26"
            fontWeight="bold"
            fill="white"
          >
            {p.toFixed(0)}%
          </text>
          <text
            x="60"
            y="74"
            textAnchor="middle"
            dominantBaseline="central"
            fontSize="12"
            fill="#9CA3AF"
          >
            Updated
          </text>
        </svg>
      </div>
    </div>
  );
};

export default WeightsUpdatedGauge;
