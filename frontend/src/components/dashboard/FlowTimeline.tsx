import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { KeystrokeEvent } from '../../types';

interface FlowTimelineProps {
  keystrokeEvents: KeystrokeEvent[];
}

const FlowTimeline: React.FC<FlowTimelineProps> = ({ keystrokeEvents }) => {
  // Process keystroke events to calculate latency and mark errors
  const chartData = useMemo(() => {
    if (keystrokeEvents.length === 0) return [];

    const processedData = [];
    let prevTimestamp = keystrokeEvents[0].timestamp;

    for (let i = 0; i < keystrokeEvents.length; i++) {
      const current = keystrokeEvents[i];
      const latency = current.timestamp - prevTimestamp; // Latency in ms
      prevTimestamp = current.timestamp;

      processedData.push({
        index: i,
        latency: latency > 0 ? latency : 0, // Ensure non-negative latency
        isError: !current.isCorrect || current.isBackspace, // Mark errors and backspaces
      });
    }
    return processedData;
  }, [keystrokeEvents]);

  // Custom Tooltip to display more info
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-container p-2 rounded-md shadow-lg text-sm text-text-light">
          <p className="font-bold">Key Index: {data.index}</p>
          <p>Latency: {data.latency} ms</p>
          {data.isError && <p className="text-error">Error/Backspace</p>}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-container p-6 rounded-xl shadow-lg h-80 flex flex-col">
      <h3 className="text-xl font-semibold text-text mb-4">Flow Timeline</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="index" hide />
          <YAxis stroke="#999" />
          <Tooltip content={<CustomTooltip />} />
          <Line 
            type="monotone" 
            dataKey="latency" 
            stroke="#8884d8" 
            strokeWidth={2} 
            dot={({ cx, cy, payload }) => {
              if (payload.isError) {
                return <circle cx={cx} cy={cy} r={5} fill="#ef4444" stroke="white" strokeWidth={1} />; // Red dot for errors
              }
              return null; // No dot for correct keys
            }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-subtle text-sm mt-2 text-center">Latency per keystroke with error markers.</p>
    </div>
  );
};

export default FlowTimeline;
