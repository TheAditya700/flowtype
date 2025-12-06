import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface GraphWidgetProps {
  data: { name: string | number; wpm: number; accuracy: string }[];
}

const GraphWidget: React.FC<GraphWidgetProps> = ({ data }) => {
  return (
    <div className="h-64 w-full min-w-[300px] flex flex-col">
      <div className="text-subtle text-xs uppercase tracking-widest mb-4">Trend</div>
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="name" hide />
            <YAxis hide domain={['dataMin - 10', 'dataMax + 10']} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'var(--container-bg)', border: 'none', borderRadius: '8px' }}
              itemStyle={{ color: 'var(--text-color)' }}
            />
            <Line type="monotone" dataKey="wpm" stroke="var(--primary-color)" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default GraphWidget;
