import React from 'react';
import Heatmap from './Heatmap';
import { RotateCcw, Play } from 'lucide-react';
import { KeystrokeEvent } from '../types';

interface BottomControlsProps {
  keystrokes: KeystrokeEvent[];
  onReset: () => void;
  onResume: () => void;
}

const BottomControls: React.FC<BottomControlsProps> = ({ keystrokes, onReset, onResume }) => {
  return (
    <div className="flex flex-col items-center gap-6 w-full">
      <div className="transform scale-90 origin-bottom">
        <Heatmap keystrokes={keystrokes} />
      </div>

      <div className="flex gap-6">
        <button 
          onClick={onReset}
          className="flex items-center gap-2 px-6 py-2 rounded-lg bg-error/10 text-error hover:bg-error/20 transition font-bold"
        >
          <RotateCcw size={18} /> Reset
        </button>
        <button 
          onClick={onResume}
          className="flex items-center gap-2 px-8 py-2 rounded-lg bg-primary text-bg hover:opacity-90 transition font-bold shadow-lg shadow-primary/20"
        >
          <Play size={18} /> Resume
        </button>
      </div>
      
      <div className="text-subtle text-xs">
        <span className="bg-container px-1 rounded text-text">Enter</span> to pause/resume
      </div>
    </div>
  );
};

export default BottomControls;
