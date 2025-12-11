import React, { useEffect, useMemo, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { TrendingUp, Target, Award, Calendar, Zap, Activity, LineChart } from 'lucide-react';
import { fetchUserStatsDetail } from '../api/client';
import { UserStatsDetail } from '../types';
import { getUserId } from '../utils/anonymousUser';
import {
  LineChart as ReLineChart,
  Line,
  Tooltip,
  CartesianGrid,
  XAxis,
  YAxis,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import CalendarHeatmap from 'react-calendar-heatmap';
import type { CalendarHeatmapValue } from 'react-calendar-heatmap';
import 'react-calendar-heatmap/dist/styles.css';
import KeyboardHeatmap from '../components/dashboard/KeyboardHeatmap';

const formatMinutes = (seconds: number) => Math.floor((seconds || 0) / 60);
const formatDateTick = (value: number) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
const formatTooltipLabel = (value: number) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
const heatmapColor = (count: number) => {
  if (count >= 4) return '#10b981'; // emerald-500
  if (count >= 3) return '#34d399'; // emerald-400
  if (count >= 2) return '#6ee7b7'; // emerald-300
  if (count >= 1) return '#a7f3d0'; // emerald-200
  return '#1f2937'; // gray-800
};

const StatsPage: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [detail, setDetail] = useState<UserStatsDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartMode, setChartMode] = useState<'wpm' | 'accuracy'>('wpm');

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const uid = getUserId();
        const res = await fetchUserStatsDetail(uid);
        setDetail(res);
      } catch (err: any) {
        setError(err.message || 'Failed to load stats');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const wpmSeries = useMemo(() => detail?.timeseries || [], [detail]);

  const chartData = useMemo(
    () =>
      wpmSeries.map(p => ({
        timestamp: p.timestamp,
        wpm: p.wpm,
        ema_wpm: p.ema_wpm ?? p.wpm,
        accuracy: p.accuracy,
        ema_accuracy: p.ema_accuracy ?? p.accuracy,
      })),
    [wpmSeries]
  );

  if (loading) {
    return (
      <div className="w-full max-w-7xl mx-auto p-6">
        <div className="text-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="text-gray-400 mt-4 font-mono">Loading stats...</p>
        </div>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="w-full max-w-7xl mx-auto p-6">
        <div className="text-center py-20">
          <Activity size={48} className="mx-auto text-gray-600 mb-4" />
          <h2 className="text-2xl font-bold text-gray-300 mb-2">No Stats Yet</h2>
          <p className="text-gray-500">Start typing to build your profile, then come back.</p>
          {error && <p className="text-red-400 mt-2">{error}</p>}
        </div>
      </div>
    );
  }

  const { summary, activity, current_streak, longest_streak } = detail;
  const charHeatmap = detail.char_heatmap || {};

  const today = new Date();
  const heatmapStart = new Date(today);
  heatmapStart.setDate(today.getDate() - 104); // Show a bit more history
  const activityMap = new Map(activity.map(a => [a.date, a.count]));
  const heatmapValues = Array.from({ length: 105 }).map((_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() - (104 - i));
    const dateStr = d.toISOString().slice(0, 10);
    return { date: dateStr, count: activityMap.get(dateStr) || 0 };
  });

  return (
    <div className="w-full max-w-[1600px] mx-auto p-6 flex flex-col gap-6">
      
      {/* Header */}
      <div className="flex items-center gap-4 pb-2 border-b border-gray-800">
        <div className="p-3 bg-blue-500/10 rounded-xl">
            <LineChart className="text-blue-500" size={28} />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Your Statistics</h1>
          <p className="text-gray-400">Track speed, consistency, and activity over time</p>
        </div>
      </div>

      {/* Personal Bests - Styled like mini widgets */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[15, 30, 60, 120].map((mode, idx) => {
            const bestWpm = Math.round((summary[`best_wpm_${mode === 15 ? '15' : mode}` as keyof typeof summary] as number) ?? 0);
            const colors = ['text-emerald-400', 'text-blue-400', 'text-purple-400', 'text-yellow-400'];
            return (
                <div key={mode} className="bg-gray-900 p-6 rounded-xl border border-gray-800 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Award size={64} className="text-white" />
                    </div>
                    <div className="text-gray-500 text-sm font-mono mb-2 uppercase tracking-wider">{mode} Seconds Best</div>
                    <div className={`text-5xl font-bold font-mono ${colors[idx % colors.length]}`}>
                        {bestWpm}
                    </div>
                    <div className="text-gray-600 text-xs font-mono mt-1">WPM</div>
                </div>
            );
        })}
      </div>

      {/* Overview Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Total Sessions */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-4">
                <div className="text-gray-500 text-sm font-mono uppercase tracking-wider">Total Sessions</div>
                <Calendar className="text-gray-600" size={20} />
            </div>
            <div className="text-4xl font-bold text-white font-mono">{summary.total_sessions}</div>
        </div>

        {/* Avg WPM */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-4">
                <div className="text-gray-500 text-sm font-mono uppercase tracking-wider">Average WPM</div>
                <TrendingUp className="text-blue-500" size={20} />
            </div>
            <div className="text-4xl font-bold text-blue-400 font-mono">{summary.avg_wpm.toFixed(0)}</div>
        </div>

        {/* Avg Accuracy */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-4">
                <div className="text-gray-500 text-sm font-mono uppercase tracking-wider">Average Accuracy</div>
                <Target className="text-emerald-500" size={20} />
            </div>
            <div className="text-4xl font-bold text-emerald-400 font-mono">{summary.avg_accuracy.toFixed(0)}<span className="text-2xl text-emerald-600">%</span></div>
        </div>

        {/* Total Time */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-4">
                <div className="text-gray-500 text-sm font-mono uppercase tracking-wider">Time Typed</div>
                <Zap className="text-yellow-500" size={20} />
            </div>
            <div className="text-4xl font-bold text-yellow-400 font-mono">{formatMinutes(summary.total_time_typing)}<span className="text-xl text-yellow-600 ml-1">min</span></div>
        </div>
      </div>

      {/* Main Chart Section */}
      <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
        <div className="flex items-center justify-between gap-3 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
               <TrendingUp className="text-blue-500" size={20} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Performance History</h2>
              <p className="text-xs text-gray-500 font-mono">Toggle between speed and accuracy</p>
            </div>
          </div>
          <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1 text-xs font-mono border border-gray-700">
            <button
              onClick={() => setChartMode('wpm')}
              className={`px-3 py-1 rounded-md transition-colors ${chartMode === 'wpm' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-200'}`}
            >
              WPM
            </button>
            <button
              onClick={() => setChartMode('accuracy')}
              className={`px-3 py-1 rounded-md transition-colors ${chartMode === 'accuracy' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-200'}`}
            >
              Accuracy
            </button>
          </div>
        </div>
        
        {chartData.length > 1 ? (
          <div className="h-80 w-full">
            <ResponsiveContainer>
              <ReLineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} vertical={false} />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDateTick}
                  tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
                  stroke="#374151"
                  tickMargin={10}
                />
                <YAxis
                  tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
                  stroke="#374151"
                  domain={chartMode === 'wpm' ? ['dataMin - 5', 'auto'] : [0, 100]}
                  tickMargin={10}
                  tickFormatter={chartMode === 'wpm' ? undefined : (v) => `${v}%`}
                />
                <Tooltip
                  labelFormatter={formatTooltipLabel}
                  formatter={(value: number, name) => {
                    if (chartMode === 'accuracy') {
                      const label = name === 'ema_accuracy' ? 'Accuracy EMA' : 'Accuracy';
                      return [`${Math.round(value as number)}%`, label];
                    }
                    return [Math.round(value as number), name === 'ema_wpm' ? 'Trend (EMA)' : 'WPM'];
                  }}
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#F3F4F6', borderRadius: '0.75rem', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                  itemStyle={{ fontFamily: 'monospace' }}
                  labelStyle={{ color: '#9CA3AF', marginBottom: '0.5rem', fontFamily: 'monospace', fontSize: '0.75rem' }}
                  cursor={{ stroke: '#4B5563', strokeWidth: 1 }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px', fontFamily: 'monospace', fontSize: '12px' }} />
                {chartMode === 'wpm' ? (
                  <>
                    <Line
                      type="monotone"
                      dataKey="wpm"
                      name="WPM"
                      stroke="#a855f7"
                      strokeWidth={2}
                      dot={{ r: 2, fill: '#a855f7', strokeWidth: 0 }}
                      activeDot={{ r: 6, stroke: '#7c3aed', strokeWidth: 2 }}
                      strokeOpacity={0.6}
                    />
                    <Line
                      type="monotone"
                      dataKey="ema_wpm"
                      name="Trend"
                      stroke="#fbbf24"
                      strokeWidth={3}
                      dot={false}
                      strokeDasharray="6 3"
                    />
                  </>
                ) : (
                  <>
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      name="Accuracy"
                      stroke="#10B981"
                      strokeWidth={3}
                      dot={{ r: 2, fill: '#10B981', strokeWidth: 0 }}
                      activeDot={{ r: 6, stroke: '#047857', strokeWidth: 2 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="ema_accuracy"
                      name="Accuracy EMA"
                      stroke="#3B82F6"
                      strokeWidth={3}
                      dot={false}
                      strokeDasharray="6 3"
                    />
                  </>
                )}
              </ReLineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500 font-mono text-sm border-2 border-dashed border-gray-800 rounded-lg">
            Complete a few sessions to unlock your history graph
          </div>
        )}
      </div>

      {/* Activity Heatmap */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Calendar Heatmap (1/3) */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 lg:col-span-1">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                        <Activity className="text-emerald-500" size={20} />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-white">Activity Log</h2>
                        <p className="text-xs text-gray-500 font-mono">Last ~3 months</p>
                    </div>
                </div>
                <div className="flex gap-3 text-xs font-mono">
                    <div className="px-3 py-1 bg-gray-800 rounded-md border border-gray-700 text-gray-400">
                        Streak: <span className="text-white font-bold">{current_streak}</span>
                    </div>
                    <div className="px-3 py-1 bg-gray-800 rounded-md border border-gray-700 text-gray-400">
                        Best: <span className="text-white font-bold">{longest_streak}</span>
                    </div>
                </div>
            </div>
            
            <div className="overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
            <div className="scale-x-100 origin-left">
                <CalendarHeatmap
                    startDate={heatmapStart}
                    endDate={today}
                    values={heatmapValues}
                    gutterSize={2}
                    showWeekdayLabels={false}
                    classForValue={(value) => {
                        if (!value) return 'color-empty';
                        return `color-scale-${value.count}`;
                    }}
                    tooltipDataAttrs={(value: any) => ({
                    title: value && value.date ? `${value.date}: ${value.count} session(s)` : 'No activity',
                    })}
                    transformDayElement={(rect: React.ReactElement, value: CalendarHeatmapValue) =>
                    React.cloneElement(rect, {
                        rx: 2,
                        ry: 2,
                        width: 9,
                        height: 9,
                        style: {
                        fill: heatmapColor(value?.count ?? 0),
                        transition: 'fill 200ms ease',
                        opacity: value?.count ? 1 : 0.4,
                        },
                    })
                    }
                />
            </div>
            </div>
            
            <div className="flex items-center justify-end gap-2 text-[10px] text-gray-500 mt-4 font-mono uppercase tracking-wide">
                <span>Less</span>
                {[0,1,2,3,4].map(v => (
                <span key={v} className="w-3 h-3 rounded-sm" style={{ backgroundColor: heatmapColor(v), opacity: v === 0 ? 0.4 : 1 }}></span>
                ))}
                <span>More</span>
            </div>
        </div>

            {/* Keyboard Heatmap (2/3) */}
            <div className="lg:col-span-2">
              <KeyboardHeatmap charStats={charHeatmap} />
            </div>
      </div>
    </div>
  );
};

export default StatsPage;