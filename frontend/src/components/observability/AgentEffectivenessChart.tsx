import React from "react";
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
import { AgentEffectivenessPoint } from "../../types";
import { Scale, buildSessionAxis, formatSessionTick, formatSessionLabel } from "../../utils/chartUtils";

interface Props {
  data: AgentEffectivenessPoint[];
  loading: boolean;
  scale: Scale;
}

const AgentEffectivenessChart: React.FC<Props> = ({ data, loading, scale }) => {
  if (loading) {
    return (
      <div className="h-64 bg-gray-900 rounded-xl border border-gray-800 animate-pulse flex items-center justify-center">
        <span className="text-gray-600 font-mono">Loading Agent Data...</span>
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
        <span className="text-gray-500 font-mono">No agent effectiveness data available</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col h-full">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white">Agent Effectiveness</h3>
        <p className="text-xs text-gray-400 font-mono">
          Mean Reward & Stability (RL Objective)
        </p>
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
            <Line
              type="monotone"
              dataKey="mean_reward"
              name="Mean Reward"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
             <Line
              type="monotone"
              dataKey="reward_std"
              name="Reward Std Dev"
              stroke="#EC4899"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AgentEffectivenessChart;
