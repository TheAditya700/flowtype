import React, { useEffect, useMemo, useState } from 'react';
import { Trophy, Timer, Target, Sparkles, AlertTriangle } from 'lucide-react';
import { useSessionMode } from '../context/SessionModeContext';
import { useAuth } from '../context/AuthContext';
import { fetchLeaderboard } from '../api/client';
import { LeaderboardEntry } from '../types';
import { getUserId } from '../utils/anonymousUser';

type Mode = '15' | '30' | '60' | '120';

const LeaderboardPage: React.FC = () => {
  const { sessionMode, setSessionMode } = useSessionMode();
  const { user } = useAuth();
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [excludeAnon, setExcludeAnon] = useState<boolean>(false);

  const modeKey: Mode = ['15', '30', '60', '120'].includes(sessionMode) ? (sessionMode as Mode) : '60';

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchLeaderboard(modeKey, excludeAnon);
        if (isMounted) setEntries(data);
      } catch (e: any) {
        if (isMounted) setError(e.message || 'Failed to load leaderboard');
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    load();
    return () => { isMounted = false; };
  }, [modeKey, excludeAnon]);

  const yourId = user?.id || getUserId();
  const yourEntryIndex = useMemo(() => entries.findIndex(e => e.user_id === yourId), [entries, yourId]);
  const yourRank = yourEntryIndex >= 0 ? yourEntryIndex + 1 : entries.length > 0 ? entries.length + 1 : null;

  return (
    <div className="w-full max-w-[1600px] mx-auto p-6 flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4 pb-2 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-500/10 rounded-xl">
            <Trophy className="text-yellow-300" size={28} />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">Leaderboard</h1>
            <p className="text-gray-400 text-sm">Timed modes only — all-time rankings.</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {(['15','30','60','120'] as Mode[]).map(mode => (
          <button
            key={mode}
            onClick={() => setSessionMode(mode)}
            className={`
              rounded-xl border p-4 text-left transition-all duration-200
              ${sessionMode === mode
                ? 'border-blue-500 bg-blue-500/10 text-white shadow-lg shadow-blue-500/10'
                : 'border-gray-800 text-gray-400 hover:border-gray-700 hover:text-white bg-gray-900'}
            `}
          >
            <div className="flex items-center gap-2 mb-1">
              <Timer size={16} className={`${sessionMode === mode ? 'text-blue-400' : 'text-gray-500'}`} />
              <span className="font-bold text-lg">{mode}s</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">Top runs in {mode}s mode</p>
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <div className="text-xs text-gray-500 font-mono">All-time · best WPM per mode</div>
        <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1 text-xs font-mono border border-gray-700">
          <button
            onClick={() => setExcludeAnon(false)}
            className={`px-3 py-1 rounded-md transition-colors ${!excludeAnon ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-200'}`}
          >
            Show anon
          </button>
          <button
            onClick={() => setExcludeAnon(true)}
            className={`px-3 py-1 rounded-md transition-colors ${excludeAnon ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-200'}`}
          >
            Hide anon
          </button>
        </div>
      </div>

      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Sparkles size={18} className="text-amber-400" />
            <span>Showing all-time · {modeKey}s mode</span>
          </div>
          <div className="text-xs text-gray-500 font-mono">Your position: <span className="text-white font-bold">{yourRank ? `#${yourRank}` : '—'}</span></div>
        </div>

        <div className="grid grid-cols-3 px-6 py-3 text-xs uppercase tracking-wide text-gray-500 border-b border-gray-800 font-mono">
          <span>Rank</span>
          <span>Player</span>
          <span className="text-right">Best WPM</span>
        </div>

        {loading ? (
          <div className="p-8 text-center text-gray-500 text-sm font-mono">Loading leaderboard...</div>
        ) : error ? (
          <div className="p-8 flex items-center justify-center gap-2 text-red-400 text-sm font-mono">
            <AlertTriangle size={16} />
            {error}
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {entries.map((row, idx) => {
              const isYou = row.user_id === yourId;
              return (
                <div
                  key={`${row.user_id}-${idx}`}
                  className={`grid grid-cols-3 px-6 py-3 items-center transition-all duration-150
                    ${isYou ? 'bg-blue-500/10 border-l-4 border-blue-500' : 'hover:bg-gray-800'}`}
                >
                  <div className="flex items-center gap-2 text-gray-400">
                    <span className="w-6 text-sm font-bold font-mono">#{idx + 1}</span>
                    {idx < 3 && <Trophy size={16} className={idx === 0 ? 'text-yellow-300' : idx === 1 ? 'text-gray-300' : 'text-amber-600'} />}                
                  </div>
                  <div className={`font-semibold ${isYou ? 'text-white' : 'text-gray-200'}`}>{row.username || 'anon'}</div>
                  <div className="text-right text-white font-bold font-mono">{Math.round(row.best_wpm)}</div>
                </div>
              );
            })}
            {entries.length === 0 && (
              <div className="p-8 text-center text-gray-500 text-sm font-mono">No runs yet for this mode.</div>
            )}
          </div>
        )}
      </div>

      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-emerald-500/10 flex items-center justify-center">
            <Target size={20} className="text-emerald-400" />
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider font-mono">Accuracy first</p>
            <p className="text-sm text-gray-300 mt-0.5">Perfect your weak keys to build a clean foundation.</p>
          </div>
        </div>
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10 flex items-center justify-center">
            <Sparkles size={20} className="text-blue-400" />
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider font-mono">Build streaks</p>
            <p className="text-sm text-gray-300 mt-0.5">Consistency beats one lucky run—aim for steady wins.</p>
          </div>
        </div>
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-amber-500/10 flex items-center justify-center">
            <Timer size={20} className="text-amber-400" />
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider font-mono">Then go faster</p>
            <p className="text-sm text-gray-300 mt-0.5">Once smooth, push your limits in short bursts.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LeaderboardPage;