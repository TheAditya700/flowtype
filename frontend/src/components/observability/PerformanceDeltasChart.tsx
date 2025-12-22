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
  ReferenceLine,
} from "recharts";
import { PerformanceDeltaPoint } from "../../types";
import { format } from "date-fns";
import { Zap, Target, Activity } from "lucide-react";
import { Timeframe, getXAxisDomain, formatXAxisTick, generateFixedTicks } from "../../utils/chartUtils";

interface Props {
  data: PerformanceDeltaPoint[];
  loading: boolean;
  timeframe: Timeframe;
}

const PerformanceDeltasChart: React.FC<Props> = ({ data, loading, timeframe }) => {
  const [mode, setMode] = useState<"wpm" | "accuracy" | "smoothness">("wpm");

  if (loading) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Performance Deltas...</span>
      </div>
    );
  }

  // Process data for the chart based on mode
  const chartData = data.map(p => ({
    ...p,
    timestamp: new Date(p.t).getTime(),
    // Multiply by 100 if they are fractions (0-1)
    display_accuracy: p.delta_accuracy * 100,
    display_smoothness: p.delta_smoothness * 100,
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
        <span className="text-gray-500 font-mono">No performance delta data available</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Performance Deltas</h3>
          <p className="text-xs text-gray-400 font-mono">
            Improvement vs. User Baseline (Zero = Baseline)
          </p>
        </div>
        
        <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg border border-gray-700">
          <button
            onClick={() => setMode("wpm")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "wpm"
                ? "bg-blue-600 text-white shadow-lg shadow-blue-900/20"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Zap size={14} /> WPM
          </button>
          <button
            onClick={() => setMode("accuracy")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "accuracy"
                ? "bg-emerald-600 text-white shadow-lg shadow-emerald-900/20"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Target size={14} /> ACC
          </button>
           <button
            onClick={() => setMode("smoothness")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "smoothness"
                ? "bg-amber-600 text-white shadow-lg shadow-amber-900/20"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Activity size={14} /> SMTH
          </button>
        </div>
      </div>

      <div className="flex-grow min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <ReferenceLine y={0} stroke="#4B5563" strokeDasharray="3 3" />
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
              tickFormatter={(val) => mode !== "wpm" ? `${val > 0 ? "+" : ""}${val.toFixed(1)}%` : `${val > 0 ? "+" : ""}${val.toFixed(1)}`}
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
              formatter={(value: number, name: string) => {
                const valStr = mode !== "wpm" ? `${value.toFixed(2)}%` : value.toFixed(2);
                return [valStr, name];
              }}
            />
            <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "12px" }} />
            
            {mode === "wpm" && (
              <Line
                type="monotone"
                dataKey="delta_effective_wpm"
                name="Δ Effective WPM"
                stroke="#3B82F6"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            )}
            
            {mode === "accuracy" && (
              <Line
                type="monotone"
                dataKey="display_accuracy"
                name="Δ Accuracy (%)"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            )}

            {mode === "smoothness" && (
               <Line
                  type="monotone"
                  dataKey="display_smoothness"
                  name="Δ Smoothness (%)"
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

export default PerformanceDeltasChart;
