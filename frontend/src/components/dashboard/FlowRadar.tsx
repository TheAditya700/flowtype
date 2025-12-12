import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip
} from 'recharts';

interface FlowRadarProps {
  metrics: {
    smoothness: number; // 0-100 (derived from 1 - CV)
    rollover: number;   // 0-100 (rollover rate * X)
    leftFluency: number; // 0-100
    rightFluency: number; // 0-100
    crossFluency: number; // 0-100
  };
}

const FlowRadar: React.FC<FlowRadarProps> = ({ metrics }) => {
  const data = [
    { subject: 'Cross', A: metrics.crossFluency, fullMark: 100 },
    { subject: 'R-Hand', A: metrics.rightFluency, fullMark: 100 },
    { subject: 'Smoothness', A: metrics.smoothness, fullMark: 100 },
    { subject: 'Rollover', A: metrics.rollover, fullMark: 100 },
    { subject: 'L-Hand', A: metrics.leftFluency, fullMark: 100 },
  ];

  return (
    <div className="w-full bg-gray-900 rounded-xl p-4 border border-gray-800 flex flex-col items-center justify-center relative min-h-[300px]">
      <h3 className="text-gray-400 text-sm font-medium absolute top-6 left-6">Flow Radar</h3>
      <ResponsiveContainer width="100%" height={385}>
        <RadarChart cx="50%" cy="53%" outerRadius="60%" data={data}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            name="Flow"
            dataKey="A"
            stroke="#8B5CF6"
            strokeWidth={3}
            fill="#8B5CF6"
            fillOpacity={0.4}
          />
          <Tooltip 
             contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
             itemStyle={{ color: '#F3F4F6' }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FlowRadar;
