import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { LearningHealthPoint } from "../../types";
import { format } from "date-fns";
import { Timeframe, getXAxisDomain, formatXAxisTick, generateFixedTicks } from "../../utils/chartUtils";
import { Eye, TrendingUp, Activity } from "lucide-react";

interface Props {
  data: LearningHealthPoint[];
  loading: boolean;
  timeframe: Timeframe;
}

const LearningHealthChart: React.FC<Props> = ({ data, loading, timeframe }) => {
  const [mode, setMode] = useState<"precision" | "variance">("precision");

  if (loading) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Health Data...</span>
      </div>
    );
  }

  // Pre-process data to ensure timestamps are numbers for Recharts domain
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
        <span className="text-gray-500 font-mono">No learning health data available</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Learning Health</h3>
          <p className="text-xs text-gray-400 font-mono">
            Bayesian Convergence
          </p>
        </div>
        
        <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg border border-gray-700">
          <button
            onClick={() => setMode("precision")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "precision"
                ? "bg-emerald-600 text-white shadow-lg shadow-emerald-900/20"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Eye size={14} /> Precision
          </button>
          <button
            onClick={() => setMode("variance")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "variance"
                ? "bg-amber-600 text-white shadow-lg shadow-amber-900/20"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Activity size={14} /> Variance
          </button>
        </div>
      </div>

      <div className="flex-grow min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={filteredData}>
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
              stroke="#4B5563"
              tick={{ fill: "#9CA3AF", fontSize: 10, fontFamily: "monospace" }}
              domain={["auto", "auto"]}
              tickFormatter={(val) => val.toFixed(4)}
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
            
            {mode === "precision" ? (
              <Line
                type="monotone"
                dataKey="mean_precision"
                name="Precision"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            ) : (
              <Line
                type="monotone"
                dataKey="mean_variance"
                name="Variance"
                stroke="#F59E0B"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default LearningHealthChart;
