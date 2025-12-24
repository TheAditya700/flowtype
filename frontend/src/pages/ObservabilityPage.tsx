import React, { useEffect, useState } from "react";
import {
  fetchObservabilityHeader,
  fetchLearningHealth,
  fetchAgentEffectiveness,
  fetchPerformanceDeltas,
  fetchUserSkillsAll,
  fetchLearningActivity,
} from "../api/client";
import {
  ObservabilityHeader as HeaderData,
  LearningHealthPoint,
  AgentEffectivenessPoint,
  PerformanceDeltaPoint,
  UserSkill,
  LearningActivityPoint,
} from "../types";
import ObservabilityHeader from "../components/observability/ObservabilityHeader";
import LearningHealthChart from "../components/observability/LearningHealthChart";
import AgentEffectivenessChart from "../components/observability/AgentEffectivenessChart";
import PerformanceDeltasChart from "../components/observability/PerformanceDeltasChart";
import LearningActivityChart from "../components/observability/LearningActivityChart";
import WeightsUpdatedGauge from "../components/observability/WeightsUpdatedGauge";
import FeatureImportanceWidget from "../components/observability/FeatureImportanceWidget";
import { Clock } from "lucide-react";
import { Scale, getScaleLimit } from "../utils/chartUtils"; // Use shared type

const ObservabilityPage: React.FC = () => {
  // State
  const [scale, setScale] = useState<Scale>("single");
  const [skillsMode, setSkillsMode] = useState<"importance" | "certain" | "uncertain">("importance");
  const [loading, setLoading] = useState(true);
  
  // Data
  const [headerData, setHeaderData] = useState<HeaderData | null>(null);
  const [learningHealth, setLearningHealth] = useState<LearningHealthPoint[]>([]);
  const [agentEffectiveness, setAgentEffectiveness] = useState<AgentEffectivenessPoint[]>([]);
  const [performanceDeltas, setPerformanceDeltas] = useState<PerformanceDeltaPoint[]>([]);
  const [userSkills, setUserSkills] = useState<UserSkill[]>([]);
  const [userSkillsAll, setUserSkillsAll] = useState<{ impact: UserSkill[]; certain: UserSkill[]; uncertain: UserSkill[] }>({ impact: [], certain: [], uncertain: [] });
  const [learningActivity, setLearningActivity] = useState<LearningActivityPoint[]>([]);

  // removed legacy compatibility function; we fetch all lists once and switch locally

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const limit = getScaleLimit(scale);
        const [
          header,
          health,
          agent,
          deltas,
          skillsAll,
          activity
        ] = await Promise.all([
          fetchObservabilityHeader(),
          fetchLearningHealth(scale, limit),
          fetchAgentEffectiveness(scale, limit),
          fetchPerformanceDeltas(scale, limit),
          fetchUserSkillsAll(10),
          fetchLearningActivity(scale, limit),
        ]);

        setHeaderData(header);
        setLearningHealth(health.points);
        setAgentEffectiveness(agent.points);
        setPerformanceDeltas(deltas.points);
        setUserSkillsAll(skillsAll);
        // pick the right list based on current mode
        const modeKey = skillsMode === "importance" ? "impact" : skillsMode;
        setUserSkills(skillsAll[modeKey]);
        setLearningActivity(activity.points);
      } catch (err) {
        console.error("Failed to load observability data:", err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
    
    // Auto-refresh every 60s
    const interval = setInterval(loadData, 60000);
    return () => clearInterval(interval);
  }, [scale]);

  // Update visible list when mode changes without refetching
  useEffect(() => {
    const modeKey = skillsMode === "importance" ? "impact" : skillsMode;
    setUserSkills(userSkillsAll[modeKey] || []);
  }, [skillsMode, userSkillsAll]);

  return (
    <div className="w-full max-w-[1600px] mx-auto p-6 space-y-6">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">System Observability</h1>
          <p className="text-gray-400 mt-1">
            Real-time insights into model learning dynamics and agent performance.
          </p>
        </div>
        
        {/* Scale Selector */}
        <div className="flex items-center gap-2 bg-gray-800 p-1 rounded-lg border border-gray-800 self-start md:self-auto">
          <Clock size={16} className="text-gray-500 ml-2" />
          {(["single", "x10", "x100", "x1000"] as Scale[]).map((option) => (
            <button
              key={option}
              onClick={() => setScale(option)}
              className={`px-3 py-1.5 rounded-md text-xs font-mono transition-colors ${
                scale === option
                  ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {option === "single" ? "1x" : option.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* KPI Cards */}
      <ObservabilityHeader data={headerData} loading={loading} />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Row 1: Health & Agent (2/3 width) and Skills (1/3 width) */}
        <div className="lg:col-span-8 grid grid-cols-1 gap-6">
            <div className="h-[350px]">
                <AgentEffectivenessChart data={agentEffectiveness} loading={loading} scale={scale} />
                 
            </div>
            <div className="h-[350px]">
                <PerformanceDeltasChart data={performanceDeltas} loading={loading} scale={scale} />
            </div>
        </div>
        
        <div className="lg:col-span-4 h-[725px]">
             <FeatureImportanceWidget 
               skills={userSkills} 
               loading={loading} 
               onModeChange={(mode) => setSkillsMode(mode)}
             />
        </div>

        {/* Row 2: Health, Weights Updated Gauge, Activity */}
        <div className="lg:col-span-6 h-[350px]">
          <LearningHealthChart data={learningHealth} loading={loading} scale={scale} />
        </div>
        <div className="lg:col-span-2 h-[350px]">
          <WeightsUpdatedGauge
            loading={loading}
            percent={
              learningActivity && learningActivity.length > 0
                ? Math.max(0, Math.min(100, (learningActivity[learningActivity.length - 1].fraction_weights_updated || 0) * 100))
                : 0
            }
          />
        </div>
        <div className="lg:col-span-4 h-[350px]">
          <LearningActivityChart data={learningActivity} loading={loading} scale={scale} />
        </div>

      </div>
    </div>
  );
};

export default ObservabilityPage;
1