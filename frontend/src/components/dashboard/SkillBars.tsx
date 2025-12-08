import React, { useMemo } from 'react';
import { UserProfile } from '../../types';

interface SkillBarsProps {
  userProfile: UserProfile | null;
}

const SkillBar: React.FC<{ label: string; value: number; max: number; colorClass: string }> = ({ label, value, max, colorClass }) => {
  const percentage = (value / max) * 100;
  return (
    <div className="mb-3">
      <div className="flex justify-between items-center text-sm mb-1">
        <span className="text-text-light">{label}</span>
        <span className="text-subtle">{value.toFixed(1)} / {max}</span>
      </div>
      <div className="w-full bg-bg rounded-full h-2.5">
        <div 
          className={`h-2.5 rounded-full ${colorClass}`} 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
};

const SkillBars: React.FC<SkillBarsProps> = ({ userProfile }) => {
  // Extract and normalize features from userProfile.features
  const skillData = useMemo(() => {
    if (!userProfile || !userProfile.features) return [];

    const features = userProfile.features;
    // These are example mappings to skill bars.
    // The actual mapping needs to be refined based on the real feature structure.
    return [
      { label: "WPM Consistency", value: features.rollingWpm || 0, max: 100, color: "bg-green-500" },
      { label: "Accuracy", value: features.rollingAccuracy ? features.rollingAccuracy * 100 : 0, max: 100, color: "bg-blue-500" },
      { label: "Backspace Control", value: features.backspaceRate ? 100 - (features.backspaceRate * 100) : 100, max: 100, color: "bg-red-500" }, // Lower rate is better
      { label: "Hesitation Control", value: features.hesitationCount ? 100 - (features.hesitationCount * 2) : 100, max: 100, color: "bg-yellow-500" }, // Lower count is better
      { label: "Flow Stability", value: features.flowScore || 0, max: 10, color: "bg-purple-500" }, // Assuming a score out of 10
      // Add more features as they become defined
    ];
  }, [userProfile]);

  if (!userProfile) {
    return (
      <div className="bg-container p-6 rounded-xl shadow-lg flex flex-col justify-center items-center h-full">
        <p className="text-subtle text-lg">Login to see your Skill Profile!</p>
      </div>
    );
  }

  return (
    <div className="bg-container p-6 rounded-xl shadow-lg flex flex-col">
      <h3 className="text-xl font-semibold text-text mb-4">Skill Profile</h3>
      <div className="flex-grow">
        {skillData.map((skill, index) => (
          <SkillBar key={index} label={skill.label} value={skill.value} max={skill.max} colorClass={skill.color} />
        ))}
      </div>
    </div>
  );
};

export default SkillBars;
