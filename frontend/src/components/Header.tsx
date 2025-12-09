import React from 'react';
import { Clock } from 'lucide-react';
import { useSessionMode } from '../context/SessionModeContext';
import UserMenu from './UserMenu'; // Import UserMenu

interface HeaderProps {
  isPaused: boolean;
  sessionStarted: boolean;
}

const Header: React.FC<HeaderProps> = ({ isPaused, sessionStarted }) => {
  const { sessionMode, setSessionMode } = useSessionMode();

  // Show header when paused OR when session hasn't started yet
  const isVisible = isPaused || !sessionStarted;

  return (
    <header className={`w-full flex justify-between items-start mb-4 z-20 relative h-12 transition-all duration-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
      <div className="flex items-center gap-4">
        <h1 className="text-3xl font-bold text-subtle tracking-tighter">nerdtype</h1>
      </div>

      <div className={`flex gap-4 transition-all duration-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
        <div className="flex items-center gap-2 bg-container p-2 rounded-lg">
            <Clock size={18} className="text-text-subtle" />
            <div className="flex bg-gray-800 rounded-lg p-1 w-fit">
              <button
                onClick={() => setSessionMode('15')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '15' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                15s
              </button>
              <button
                onClick={() => setSessionMode('30')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '30' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                30s
              </button>
              <button
                onClick={() => setSessionMode('60')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '60' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                60s
              </button>
              <button
                onClick={() => setSessionMode('120')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === '120' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                120s
              </button>
              <button
                onClick={() => setSessionMode('free')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  sessionMode === 'free' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                Free
              </button>
            </div>
        </div>
        <UserMenu /> 
      </div>
    </header>
  );
};

export default Header;
