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
import { Scale, buildSessionAxis, formatSessionTick, formatSessionLabel } from "../../utils/chartUtils";
import { Eye, Activity } from "lucide-react";

interface Props {
  data: LearningHealthPoint[];
  loading: boolean;
  scale: Scale;
}

const LearningHealthChart: React.FC<Props> = ({ data, loading, scale }) => {
  const [mode, setMode] = useState<"precision" | "variance">("precision");

  if (loading) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Health Data...</span>
      </div>
    );
  }

  const chartData = data.map((d, idx) => ({
    ...d,
    session: idx + 1,
  }));

  const { domain, ticks } = buildSessionAxis(chartData.length);

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
        
        <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg border border-gray-800">
          <button
            onClick={() => setMode("precision")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "precision"
                ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Eye size={14} /> Precision
          </button>
          <button
            onClick={() => setMode("variance")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              mode === "variance"
                ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Activity size={14} /> Variance
          </button>
        </div>
      </div>

      <div className="flex-grow min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
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
              labelFormatter={(value) => formatSessionLabel(value as number, scale)}
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
