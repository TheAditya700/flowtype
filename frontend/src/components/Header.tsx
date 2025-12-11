import React, { useState } from 'react';
import { Clock, Keyboard, BarChart3, BookOpen, Trophy, User, LogOut } from 'lucide-react';
import { useSessionMode } from '../context/SessionModeContext';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

interface HeaderProps {
  isPaused: boolean;
  sessionStarted: boolean;
}

const Header: React.FC<HeaderProps> = ({ isPaused, sessionStarted }) => {
  const { sessionMode, setSessionMode } = useSessionMode();
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, logout, user } = useAuth();
  const [showDropdown, setShowDropdown] = useState(false);

  // Show header when paused OR when session hasn't started yet (only on type page)
  const isTypePage = location.pathname === '/';
  const isLeaderboardPage = location.pathname === '/leaderboard';
  const isVisible = !isTypePage || isPaused || !sessionStarted;

  const isActive = (path: string) => location.pathname === path;

  const handleAuthTab = async () => {
    if (isAuthenticated) {
      await logout();
      navigate('/');
    } else {
      navigate('/auth');
    }
  };

  const disableModeSwitch = isTypePage && sessionStarted;

  return (
    <header className={`w-full flex items-center justify-between mb-4 z-20 relative h-16 transition-all duration-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
      {/* Left: Logo */}
      <div className="flex items-center gap-3">
        <h1
          className="text-3xl font-bold text-subtle tracking-tighter cursor-pointer hover:text-text transition"
          onClick={() => navigate('/')}
        >
          nerdtype
        </h1>
      </div>

      {/* Center: Tabs absolutely centered to viewport */}
      <div className="absolute left-1/2 -translate-x-1/2 flex flex-col items-center gap-2">
        <nav className="flex items-center gap-1 bg-container rounded-lg p-1">
          <button
            onClick={() => navigate('/')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
              isActive('/') ? 'bg-primary text-bg font-semibold' : 'text-text-subtle hover:text-text hover:bg-gray-700'
            }`}
          >
            <Keyboard size={16} />
            <span className="text-sm">type</span>
          </button>

          <button
            onClick={() => navigate('/stats')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
              isActive('/stats') ? 'bg-primary text-bg font-semibold' : 'text-text-subtle hover:text-text hover:bg-gray-700'
            }`}
          >
            <BarChart3 size={16} />
            <span className="text-sm">stats</span>
          </button>

          <button
            onClick={() => navigate('/leaderboard')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
              isActive('/leaderboard') ? 'bg-primary text-bg font-semibold' : 'text-text-subtle hover:text-text hover:bg-gray-700'
            }`}
          >
            <Trophy size={16} />
            <span className="text-sm">leaderboard</span>
          </button>

          <button
            onClick={() => navigate('/wiki')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
              isActive('/wiki') ? 'bg-primary text-bg font-semibold' : 'text-text-subtle hover:text-text hover:bg-gray-700'
            }`}
          >
            <BookOpen size={16} />
            <span className="text-sm">wiki</span>
          </button>
        </nav>

      </div>

      {/* Right: Selector + Auth */}
      <div className="flex items-center gap-3 justify-end ml-auto">
        {(isTypePage || isLeaderboardPage) && (
          <div className={`flex items-center gap-2 bg-container px-3 py-2 rounded-lg transition-all duration-500 h-12 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
            <Clock size={18} className="text-text-subtle" />
            <div className="flex bg-gray-800 rounded-lg p-1 w-fit h-8">
              <button
                disabled={disableModeSwitch}
                onClick={() => setSessionMode('15')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '15' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
                aria-disabled={disableModeSwitch}
              >
                15s
              </button>
              <button
                disabled={disableModeSwitch}
                onClick={() => setSessionMode('30')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '30' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
                aria-disabled={disableModeSwitch}
              >
                30s
              </button>
              <button
                disabled={disableModeSwitch}
                onClick={() => setSessionMode('60')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '60' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
                aria-disabled={disableModeSwitch}
              >
                60s
              </button>
              <button
                disabled={disableModeSwitch}
                onClick={() => setSessionMode('120')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '120' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
                aria-disabled={disableModeSwitch}
              >
                120s
              </button>
              {isTypePage && (
                <button
                  disabled={disableModeSwitch}
                  onClick={() => setSessionMode('free')}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    sessionMode === 'free' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                  }`}
                  aria-disabled={disableModeSwitch}
                >
                  Free
                </button>
              )}
            </div>
          </div>
        )}

        <div className="relative h-12 flex items-center">
          {isAuthenticated ? (
            <div
              className="bg-container px-3 py-2 h-full rounded-lg border border-gray-700 flex items-center gap-2 cursor-pointer"
              onClick={() => setShowDropdown(v => !v)}
            >
              <User size={18} className="text-primary" />
              <span className="text-sm font-semibold text-text">{user?.username || 'user'}</span>
            </div>
          ) : (
            <button
              onClick={() => navigate('/auth')}
              className="bg-container px-3 py-2 h-full rounded-lg border border-gray-700 flex items-center gap-2 text-text-subtle hover:text-text hover:border-primary"
            >
              <User size={18} className="text-primary" />
              <span className="text-sm font-semibold">register / login</span>
            </button>
          )}

          {showDropdown && isAuthenticated && (
            <div className="absolute right-0 top-full mt-2 w-48 bg-container border border-gray-700 rounded-lg shadow-lg z-30">
              <button
                onClick={() => { navigate('/account/change-username'); setShowDropdown(false); }}
                className="w-full text-left px-3 py-2 text-sm text-text hover:bg-gray-800"
              >
                Change username
              </button>
              <button
                onClick={() => { navigate('/account/change-password'); setShowDropdown(false); }}
                className="w-full text-left px-3 py-2 text-sm text-text hover:bg-gray-800 border-t border-gray-700"
              >
                Change password
              </button>
              <button
                onClick={() => { navigate('/account/delete'); setShowDropdown(false); }}
                className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-gray-800 border-t border-gray-700"
              >
                Delete account
              </button>
              <button
                onClick={async () => { await logout(); setShowDropdown(false); navigate('/'); }}
                className="w-full text-left px-3 py-2 text-sm text-text hover:bg-gray-800 flex items-center gap-2 border-t border-gray-700"
              >
                <LogOut size={16} />
                Logout
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
