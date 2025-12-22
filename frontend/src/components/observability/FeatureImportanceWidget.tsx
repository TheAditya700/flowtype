import React, { useState } from "react";
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

  if (skills.length === 0) {
    return (
      <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex items-center justify-center">
        <span className="text-gray-500 font-mono">No feature data available</span>
      </div>
    );
  }

  const maxImportance = Math.max(...skills.map((s) => s.importance || 0), 1);
  const maxPrecision = Math.max(...skills.map((s) => s.precision || 0), 1);

  const getModeTitle = () => {
    switch (mode) {
      case "importance": return "Top Influential Factors";
      case "certain": return "Most Certain Beliefs";
      case "uncertain": return "Most Uncertain Beliefs";
    }
  };

  const getModeDescription = () => {
    switch (mode) {
      case "importance": return "User skills driving snippet selection";
      case "certain": return "Features with highest confidence (precision)";
      case "uncertain": return "Features with lowest confidence (high variance)";
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
      <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg border border-gray-700 mb-4">
        <button
          onClick={() => handleModeChange("importance")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all flex-1 justify-center ${
            mode === "importance"
              ? "bg-blue-600 text-white shadow-lg shadow-blue-900/20"
              : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <TrendingUp size={14} /> Impact
        </button>
        <button
          onClick={() => handleModeChange("certain")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all flex-1 justify-center ${
            mode === "certain"
              ? "bg-green-600 text-white shadow-lg shadow-green-900/20"
              : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <Shield size={14} /> Certain
        </button>
        <button
          onClick={() => handleModeChange("uncertain")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all flex-1 justify-center ${
            mode === "uncertain"
              ? "bg-amber-600 text-white shadow-lg shadow-amber-900/20"
              : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <AlertCircle size={14} /> Uncertain
        </button>
      </div>

      <div className="flex-grow space-y-4 overflow-y-auto pr-2 custom-scrollbar">
        {skills.map((skill, idx) => {
          const userFeatureName = getFeatureName(skill.user_feature_idx, "user");
          
          return (
          <div key={idx} className="group">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-mono text-gray-300 truncate max-w-[70%]">
                {userFeatureName}
              </span>
              <div className="flex items-center gap-2">
                 <span className={`text-xs font-mono font-bold ${
                    skill.sign === 'positive' ? 'text-green-400' :
                    skill.sign === 'negative' ? 'text-red-400' : 'text-gray-400'
                 }`}>
                  {skill.sign === 'positive' ? <ArrowUp size={12} className="inline" /> :
                   skill.sign === 'negative' ? <ArrowDown size={12} className="inline" /> :
                   <Minus size={12} className="inline" />}
                 </span>
                 <span className="text-xs font-mono text-gray-500">
                   {mode === "importance" ? `w=${(skill.importance || 0).toFixed(4)}` : `p=${(skill.precision || 0).toFixed(2)}`}
                 </span>
              </div>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                    mode === "certain" ? 'bg-green-500' :
                    mode === "uncertain" ? 'bg-amber-500' :
                    skill.sign === 'positive' ? 'bg-green-500' :
                    skill.sign === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                }`}
                style={{ width: `${mode === "importance" ? ((skill.importance || 0) / maxImportance) * 100 : ((skill.precision || 0) / maxPrecision) * 100}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between items-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-[10px] font-mono text-gray-600">
                    Precision: {(skill.precision || 0).toFixed(2)}
                </span>
                <span className="text-[10px] font-mono text-gray-600">
                    Weight: {(skill.importance || 0).toFixed(4)}
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
