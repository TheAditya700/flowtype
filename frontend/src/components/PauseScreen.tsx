import React from 'react';
import { SnippetLog, KeystrokeEvent } from '../types';
import Heatmap from './Heatmap';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, RotateCcw } from 'lucide-react';

interface PauseScreenProps {
  onResume: () => void;
  onReset: () => void;
  logs: SnippetLog[];
  allKeystrokes: KeystrokeEvent[];
  totalWords: number;
  totalTime: number;
}

const PauseScreen: React.FC<PauseScreenProps> = ({ 
  onResume, 
  onReset, 
  logs, 
  allKeystrokes, 
  totalWords, 
  totalTime 
}) => {
  
  // Calculate net session stats
  const avgWpm = logs.length > 0 
    ? (logs.reduce((acc, l) => acc + l.wpm, 0) / logs.length).toFixed(0) 
    : 0;
  
  const avgAcc = logs.length > 0 
    ? (logs.reduce((acc, l) => acc + l.accuracy, 0) / logs.length * 100).toFixed(1) 
    : 100;

  // Prepare chart data
  const chartData = logs.map((log, idx) => ({
    name: idx + 1,
    wpm: log.wpm,
    accuracy: (log.accuracy * 100).toFixed(1),
    difficulty: log.difficulty
  }));

  return (
    <div className="fixed inset-0 bg-gray-900/95 backdrop-blur-sm z-50 flex items-center justify-center p-8">
      <div className="bg-gray-800 w-full max-w-5xl max-h-[90vh] rounded-2xl shadow-2xl overflow-y-auto flex flex-col">
        
        {/* Header */}
        <div className="p-6 border-b border-gray-700 flex justify-between items-center">
          <h2 className="text-3xl font-bold text-white">Session Paused</h2>
          <div className="flex gap-4">
             <button 
              onClick={onReset}
              className="flex items-center gap-2 px-4 py-2 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition"
            >
              <RotateCcw size={20} /> Reset
            </button>
            <button 
              onClick={onResume}
              className="flex items-center gap-2 px-6 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 transition font-bold shadow-lg shadow-blue-500/20"
            >
              <Play size={20} /> Resume
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Col: High Level Stats */}
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700/50 p-4 rounded-xl">
                <p className="text-gray-400 text-sm">Avg WPM</p>
                <p className="text-3xl font-mono text-blue-400">{avgWpm}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-xl">
                <p className="text-gray-400 text-sm">Accuracy</p>
                <p className="text-3xl font-mono text-green-400">{avgAcc}%</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-xl">
                <p className="text-gray-400 text-sm">Words Typed</p>
                <p className="text-3xl font-mono text-purple-400">{totalWords}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-xl">
                <p className="text-gray-400 text-sm">Time Active</p>
                <p className="text-3xl font-mono text-yellow-400">{(totalTime / 60).toFixed(1)}m</p>
              </div>
            </div>
            
            {/* Heatmap Component */}
            <div className="bg-gray-700/30 p-2 rounded-xl">
              <Heatmap keystrokes={allKeystrokes} />
            </div>
          </div>

          {/* Right Col: Graphs */}
          <div className="lg:col-span-2 bg-gray-700/30 p-6 rounded-xl flex flex-col">
            <h3 className="text-gray-300 font-semibold mb-4">Performance History</h3>
            
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis yAxisId="left" stroke="#60A5FA" />
                  <YAxis yAxisId="right" orientation="right" stroke="#34D399" domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                    itemStyle={{ color: '#E5E7EB' }}
                  />
                  <Line yAxisId="left" type="monotone" dataKey="wpm" stroke="#60A5FA" strokeWidth={3} dot={false} activeDot={{ r: 8 }} name="WPM" />
                  <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#34D399" strokeWidth={2} dot={false} name="Accuracy %" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-6 bg-gray-700/50 p-4 rounded-lg text-sm text-gray-400">
              <h4 className="text-gray-300 font-bold mb-2">Analysis</h4>
              <ul className="list-disc pl-4 space-y-1">
                <li>Current Difficulty Level: <span className="text-white">{logs.length > 0 ? logs[logs.length-1].difficulty.toFixed(1) : '5.0'}</span></li>
                <li>Snippets Completed: <span className="text-white">{logs.length}</span></li>
              </ul>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default PauseScreen;
