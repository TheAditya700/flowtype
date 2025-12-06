import React from 'react';
import { User, Palette } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface HeaderProps {
  isPaused: boolean;
}

const Header: React.FC<HeaderProps> = ({ isPaused }) => {
  const { theme, setTheme } = useTheme();

  return (
    <header className="w-full flex justify-end items-start mb-4 z-20 relative h-12">
      {/* Top Right: Controls (Visible when Paused) */}
      <div className={`flex gap-4 transition-all duration-500 ${isPaused ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
        <div className="flex items-center gap-2 bg-container p-2 rounded-lg">
            <Palette size={18} className="text-text-subtle" />
            <select 
                value={theme} 
                onChange={(e) => setTheme(e.target.value as any)}
                className="bg-transparent text-text outline-none text-sm font-mono cursor-pointer"
            >
                <option value="default">Flow</option>
                <option value="tokyo-night">Tokyo</option>
                <option value="catppuccin">Catppuccin</option>
                <option value="nord">Nord</option>
                <option value="gruvbox">Gruvbox</option>
            </select>
        </div>
        <button className="flex items-center gap-2 bg-container p-2 rounded-lg text-text hover:text-primary transition">
            <User size={18} />
        </button>
      </div>
    </header>
  );
};

export default Header;
