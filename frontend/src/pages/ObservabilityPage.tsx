import React, { useEffect, useState } from "react";
import {
  fetchObservabilityHeader,
  fetchLearningHealth,
  fetchAgentEffectiveness,
  fetchPerformanceDeltas,
  fetchUserSkillsImportance,
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
import FeatureImportanceWidget from "../components/observability/FeatureImportanceWidget";
import { Clock } from "lucide-react";
import { Timeframe } from "../utils/chartUtils"; // Use shared type

const ObservabilityPage: React.FC = () => {
  // State
  const [timeframe, setTimeframe] = useState<Timeframe>("day");
  const [skillsMode, setSkillsMode] = useState<"importance" | "certain" | "uncertain">("importance");
  const [loading, setLoading] = useState(true);
  
  // Data
  const [headerData, setHeaderData] = useState<HeaderData | null>(null);
  const [learningHealth, setLearningHealth] = useState<LearningHealthPoint[]>([]);
  const [agentEffectiveness, setAgentEffectiveness] = useState<AgentEffectivenessPoint[]>([]);
  const [performanceDeltas, setPerformanceDeltas] = useState<PerformanceDeltaPoint[]>([]);
  const [userSkills, setUserSkills] = useState<UserSkill[]>([]);
  const [learningActivity, setLearningActivity] = useState<LearningActivityPoint[]>([]);

  const loadUserSkills = async (mode: "importance" | "certain" | "uncertain") => {
    try {
      const skills = await fetchUserSkillsImportance(10, mode);
      setUserSkills(skills.skills);
    } catch (err) {
      console.error("Failed to load user skills:", err);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [
          header,
          health,
          agent,
          deltas,
          skills,
          activity
        ] = await Promise.all([
          fetchObservabilityHeader(),
          fetchLearningHealth(timeframe),
          fetchAgentEffectiveness(timeframe),
          fetchPerformanceDeltas(timeframe),
          fetchUserSkillsImportance(10, skillsMode),
          fetchLearningActivity(timeframe),
        ]);

        setHeaderData(header);
        setLearningHealth(health.points);
        setAgentEffectiveness(agent.points);
        setPerformanceDeltas(deltas.points);
        setUserSkills(skills.skills);
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
  }, [timeframe, skillsMode]);

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
        
        {/* Timeframe Selector */}
        <div className="flex items-center gap-2 bg-gray-900 p-1 rounded-lg border border-gray-800 self-start md:self-auto">
          <Clock size={16} className="text-gray-500 ml-2" />
          {(["hour", "day", "week", "month"] as Timeframe[]).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1.5 rounded-md text-xs font-mono transition-colors ${
                timeframe === tf
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              }`}
            >
              {tf.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* KPI Cards */}
      <ObservabilityHeader data={headerData} loading={loading} />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Row 1: Health & Agent (2/3 width) and Skills (1/3 width) */}
        <div className="lg:col-span-2 grid grid-cols-1 gap-6">
            <div className="h-[350px]">
                <LearningHealthChart data={learningHealth} loading={loading} timeframe={timeframe} />
            </div>
            <div className="h-[350px]">
                <AgentEffectivenessChart data={agentEffectiveness} loading={loading} timeframe={timeframe} />
            </div>
        </div>
        
        <div className="lg:col-span-1 h-[725px]">
             <FeatureImportanceWidget 
               skills={userSkills} 
               loading={loading} 
               onModeChange={(mode) => setSkillsMode(mode)}
             />
        </div>

        {/* Row 2: Performance Deltas & Activity */}
        <div className="lg:col-span-2 h-[350px]">
             <PerformanceDeltasChart data={performanceDeltas} loading={loading} timeframe={timeframe} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
             <LearningActivityChart data={learningActivity} loading={loading} timeframe={timeframe} />
        </div>

      </div>
    </div>
  );
};

export default ObservabilityPage;
1