import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { SpeedPoint } from '../../types';

interface SpeedGraphProps {
  data: SpeedPoint[];
}

const SpeedGraph: React.FC<SpeedGraphProps> = ({ data }) => {
  return (
    <div className="w-full h-[300px] bg-gray-900 rounded-xl p-4 border border-gray-800">
      <h3 className="text-gray-400 text-sm font-medium mb-4">Speed Graph</h3>
      <ResponsiveContainer width="94%" height="90%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.5} />
          <XAxis 
            dataKey="time" 
            stroke="#9CA3AF" 
            tick={{ fontSize: 12 }} 
            tickFormatter={(val) => `${val}s`}
          />
          <YAxis 
            stroke="#9CA3AF" 
            tick={{ fontSize: 12 }}
            domain={['auto', 'auto']}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
            itemStyle={{ color: '#F3F4F6' }}
            formatter={(value: number) => [Math.round(value), 'WPM']}
            labelFormatter={(label) => `${label}s`}
          />
          <Line 
            type="monotone" 
            dataKey="wpm" 
            stroke="#3B82F6" 
            strokeWidth={3} 
            dot={false}
            activeDot={{ r: 6 }}
          />
          <Line 
            type="monotone" 
            dataKey="rawWpm" 
            stroke="#6B7280" 
            strokeWidth={1} 
            strokeDasharray="5 5" 
            dot={false} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SpeedGraph;
