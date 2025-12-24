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
import { Zap, Target, Activity } from "lucide-react";
import { Scale, buildSessionAxis, formatSessionTick, formatSessionLabel } from "../../utils/chartUtils";

interface Props {
  data: PerformanceDeltaPoint[];
  loading: boolean;
  scale: Scale;
}

const PerformanceDeltasChart: React.FC<Props> = ({ data, loading, scale }) => {
  const [mode, setMode] = useState<"wpm" | "accuracy" | "smoothness">("wpm");

  if (loading) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Performance Deltas...</span>
      </div>
    );
  }

  // Process data for the chart based on mode
  const chartData = data.map((p, idx) => ({
    ...p,
    session: idx + 1,
    // Multiply by 100 if they are fractions (0-1)
    display_accuracy: p.delta_accuracy * 100,
    display_smoothness: p.delta_smoothness * 100,
  }));

  const { domain, ticks } = buildSessionAxis(chartData.length);

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
        
        <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg border border-gray-800">
          <button
            onClick={() => setMode("wpm")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "wpm"
                ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Zap size={14} /> WPM
          </button>
          <button
            onClick={() => setMode("accuracy")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "accuracy"
                ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Target size={14} /> ACC
          </button>
           <button
            onClick={() => setMode("smoothness")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "smoothness"
                ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Activity size={14} /> SMTH
          </button>
        </div>
      </div>

      <div className="flex-grow min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <ReferenceLine y={0} stroke="#4B5563" strokeDasharray="3 3" />
            <XAxis
              dataKey="session"
              type="number"
              domain={domain}
              ticks={ticks}
              tickFormatter={(t) => formatSessionTick(t as number, scale)}
              stroke="#4B5563"
              tick={{ fill: "#9CA3AF", fontSize: 10, fontFamily: "monospace" }}
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
              labelFormatter={(value) => formatSessionLabel(value as number, scale)}
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
