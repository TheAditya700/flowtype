import React, { useMemo, useState } from "react";
import { UserSkill } from "../../types";
import { getFeatureName } from "../../utils/featureNames";
import { ArrowUp, ArrowDown, Minus, TrendingUp, Shield, AlertCircle } from "lucide-react";

interface Props {
  skills: UserSkill[];
  loading: boolean;
  onModeChange?: (mode: "importance" | "certain" | "uncertain") => void;
}

const FeatureImportanceWidget: React.FC<Props> = ({ skills, loading, onModeChange }) => {
  const [mode, setMode] = useState<"importance" | "certain" | "uncertain">("importance");

  // Ensure hooks run in consistent order every render
  const safeSkills: UserSkill[] = Array.isArray(skills) ? skills : [];

  const maxByMode = useMemo(() => {
    try {
      switch (mode) {
        case "importance":
          return Math.max(...safeSkills.map((s) => s.impact ?? 0), 1);
        case "certain":
          return Math.max(...safeSkills.map((s) => s.certainty ?? 0), 1);
        case "uncertain":
          return Math.max(...safeSkills.map((s) => s.uncertainty ?? 0), 1);
        default:
          return 1;
      }
    } catch {
      return 1;
    }
  }, [mode, safeSkills]);

  const metricLabel = mode === "importance" ? "Impact" : mode === "certain" ? "Certain" : "Uncertain";
  const valueFor = (s: UserSkill) => (mode === "importance" ? (s.impact ?? 0) : mode === "certain" ? (s.certainty ?? 0) : (s.uncertainty ?? 0));

  const handleModeChange = (newMode: "importance" | "certain" | "uncertain") => {
    setMode(newMode);
    if (onModeChange) {
      onModeChange(newMode);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 animate-pulse">
        <div className="h-6 w-1/3 bg-gray-800 rounded mb-4" />
        <div className="space-y-3">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
            <div key={i} className="h-10 bg-gray-800 rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (safeSkills.length === 0) {
    return (
      <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex items-center justify-center">
        <span className="text-gray-500 font-mono">No feature data available</span>
      </div>
    );
  }

  const getModeTitle = () => {
    switch (mode) {
      case "importance": return "Top Influential Factors";
      case "certain": return "Most Certain Beliefs";
      case "uncertain": return "Most Uncertain Beliefs";
    }
  };

  const getModeDescription = () => {
    switch (mode) {
      case "importance": return "Impact = Σ E[|W|] across components";
      case "certain": return "Certain = mean P(|W| > ε) across components";
      case "uncertain": return "Uncertain = Σ Var(W)·E[|W|] across components";
    }
  };

  return (
    <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 h-full flex flex-col">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white">{getModeTitle()}</h3>
        <p className="text-xs text-gray-400 font-mono">
          {getModeDescription()}
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg border border-gray-800 mb-4">
        <button
          onClick={() => handleModeChange("importance")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all flex-1 justify-center ${
            mode === "importance"
              ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <TrendingUp size={14} /> Impact
        </button>
        <button
          onClick={() => handleModeChange("certain")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all flex-1 justify-center ${
            mode === "certain"
              ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <Shield size={14} /> Certain
        </button>
        <button
          onClick={() => handleModeChange("uncertain")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all flex-1 justify-center ${
            mode === "uncertain"
              ?"bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <AlertCircle size={14} /> Uncertain
        </button>
      </div>

      <div className="flex-grow space-y-4 overflow-y-auto pr-2 custom-scrollbar">
        {safeSkills.map((skill, idx) => {
          let userFeatureName = String(skill.user_feature_idx);
          try {
            const name = getFeatureName(skill.user_feature_idx, "user" as any);
            if (typeof name === 'string' && name.length > 0) userFeatureName = name;
          } catch {}
          
          const derivedSign = (skill.mean_weight ?? 0) > 0 ? 'positive' : (skill.mean_weight ?? 0) < 0 ? 'negative' : 'neutral';
          return (
          <div key={idx} className="group">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-mono text-gray-300 truncate max-w-[70%]">
                {userFeatureName}
              </span>
              <div className="flex items-center gap-2">
                 <span className={`text-xs font-mono font-bold ${
                    derivedSign === 'positive' ? 'text-green-400' :
                    derivedSign === 'negative' ? 'text-red-400' : 'text-gray-400'
                 }`}>
                  {derivedSign === 'positive' ? <ArrowUp size={12} className="inline" /> :
                   derivedSign === 'negative' ? <ArrowDown size={12} className="inline" /> :
                   <Minus size={12} className="inline" />}
                 </span>
                 <span className="text-xs font-mono text-gray-500">
                   {metricLabel[0]}={(valueFor(skill) || 0).toFixed(4)}
                 </span>
              </div>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                    mode === "certain" ? 'bg-green-500' :
                    mode === "uncertain" ? 'bg-amber-500' :
                    derivedSign === 'positive' ? 'bg-green-500' :
                    derivedSign === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                }`}
                style={{ width: `${(() => {
                  const denom = Number.isFinite(maxByMode) && maxByMode > 0 ? maxByMode : 1;
                  const raw = (valueFor(skill) / denom) * 100;
                  const pct = Math.max(0, Math.min(100, Number.isFinite(raw) ? raw : 0));
                  return pct;
                })()}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between items-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-[10px] font-mono text-gray-600">
                    {metricLabel}: {(valueFor(skill) || 0).toFixed(4)}
                </span>
                <span className="text-[10px] font-mono text-gray-600">
                    Impact: {(skill.impact || 0).toFixed(4)}
                </span>
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
};

export default FeatureImportanceWidget;
