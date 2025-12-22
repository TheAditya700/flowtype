import React from "react";
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { LearningActivityPoint } from "../../types";
import { format } from "date-fns";
import { Timeframe, getXAxisDomain, formatXAxisTick, generateFixedTicks } from "../../utils/chartUtils";

interface Props {
  data: LearningActivityPoint[];
  loading: boolean;
  timeframe: Timeframe;
}

const LearningActivityChart: React.FC<Props> = ({ data, loading, timeframe }) => {
  if (loading) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Learning Activity...</span>
      </div>
    );
  }

  // Pre-process data
  const chartData = data.map(d => ({
    ...d,
    timestamp: new Date(d.t).getTime()
  }));

  const lastTimestamp = chartData.length > 0 ? Math.max(...chartData.map(d => d.timestamp)) : undefined;
  const domain = getXAxisDomain(timeframe, lastTimestamp);
  const ticks = generateFixedTicks(timeframe, lastTimestamp);
  
  // Filter to include points within domain plus one point before for left border extension
  const [domainStart, domainEnd] = domain;
  const filteredData = chartData.filter((d, idx, arr) => {
    if (d.timestamp >= domainStart && d.timestamp <= domainEnd) return true;
    // Include one point before the domain start
    if (d.timestamp < domainStart && (idx === arr.length - 1 || arr[idx + 1].timestamp >= domainStart)) return true;
    return false;
  });

  if (data.length === 0) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center">
        <span className="text-gray-500 font-mono">No learning activity data available</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col h-full">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white">Learning Activity</h3>
        <p className="text-xs text-gray-400 font-mono">
          Weight Updates & Magnitude (Adaptation Rate)
        </p>
      </div>
      <div className="flex-grow min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={domain}
              ticks={ticks}
              tickFormatter={(t) => formatXAxisTick(t, timeframe)}
              stroke="#4B5563"
              tick={{ fill: "#9CA3AF", fontSize: 10, fontFamily: "monospace" }}
              scale="time"
            />
            <YAxis
              yAxisId="left"
              stroke="#4B5563"
              tick={{ fill: "#9CA3AF", fontSize: 10, fontFamily: "monospace" }}
              label={{ value: '% Updated', angle: -90, position: 'insideLeft', fill: '#9CA3AF', fontSize: 10 }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#4B5563"
              tick={{ fill: "#9CA3AF", fontSize: 10, fontFamily: "monospace" }}
              label={{ value: 'Mean Δ', angle: 90, position: 'insideRight', fill: '#9CA3AF', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                borderColor: "#374151",
                color: "#F3F4F6",
                borderRadius: "0.5rem",
              }}
              itemStyle={{ fontFamily: "monospace", fontSize: "12px" }}
              labelStyle={{ fontFamily: "monospace", color: "#9CA3AF", fontSize: "12px" }}
              labelFormatter={(t) => format(new Date(t), "PPpp")}
            />
            <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "12px" }} />
            <Bar
              yAxisId="left"
              dataKey="fraction_weights_updated"
              name="% Weights Updated"
              fill="#6366F1"
              barSize={20}
              fillOpacity={0.6}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="mean_abs_delta_mean"
              name="Mean Weight Delta"
              stroke="#EC4899"
              strokeWidth={2}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default LearningActivityChart;
