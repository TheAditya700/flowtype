import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { User, LogOut } from 'lucide-react';
import AuthModal from './AuthModal'; // Assuming AuthModal.tsx is in the same directory

interface UserMenuProps {
  // Add any props needed for the menu, e.g., to trigger auth modal
}

const UserMenu: React.FC<UserMenuProps> = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);

  const handleAuthClick = () => {
    if (!isAuthenticated) {
      setShowAuthModal(true);
    }
  };

  return (
    <>
      <div className="relative">
        {isAuthenticated ? (
          <div className="flex items-center gap-2 text-text">
            <User size={20} className="text-primary" />
            <span className="font-semibold">{user?.username}</span>
            <button onClick={logout} className="p-2 rounded-md hover:bg-gray-700 transition">
              <LogOut size={18} />
            </button>
          </div>
        ) : (
          <button
            onClick={handleAuthClick}
            className="flex items-center gap-2 px-3 py-2 rounded-md bg-primary text-bg font-semibold hover:opacity-90 transition"
          >
            <User size={18} /> Login / Register
          </button>
        )}
      </div>
      {showAuthModal && <AuthModal showModal={showAuthModal} onClose={() => setShowAuthModal(false)} />}
    </>
  );
};

export default UserMenu;
