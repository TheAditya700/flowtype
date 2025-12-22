import React from "react";
import { ObservabilityHeader as HeaderData } from "../../types";
import { Users, Database, GitBranch, Clock } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

interface Props {
  data: HeaderData | null;
  loading: boolean;
}

const ObservabilityHeader: React.FC<Props> = ({ data, loading }) => {
  const cards = [
    {
      label: "Active Users (24h)",
      value: data?.active_users ?? 0,
      icon: Users,
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
    },
    {
      label: "Total Sessions",
      value: data?.total_sessions.toLocaleString() ?? 0,
      icon: Database,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
    {
      label: "Model Version",
      value: data?.model_version ?? "v0.0.0",
      icon: GitBranch,
      color: "text-purple-400",
      bgColor: "bg-purple-500/10",
    },
    {
      label: "Last Snapshot",
      value: data?.last_snapshot_time
        ? formatDistanceToNow(new Date(data.last_snapshot_time), { addSuffix: true })
        : "N/A",
      icon: Clock,
      color: "text-yellow-400",
      bgColor: "bg-yellow-500/10",
    },
  ];

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-gray-900 h-24 rounded-xl border border-gray-800 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, idx) => (
        <div key={idx} className="bg-gray-900 p-4 rounded-xl border border-gray-800 flex items-center gap-4">
          <div className={`p-3 rounded-lg ${card.bgColor}`}>
            <card.icon className={card.color} size={24} />
          </div>
          <div>
            <div className="text-gray-400 text-xs font-mono uppercase tracking-wider mb-1">
              {card.label}
            </div>
            <div className="text-xl font-bold text-white font-mono">
              {card.value}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ObservabilityHeader;
