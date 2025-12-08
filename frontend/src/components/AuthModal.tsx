import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { X } from 'lucide-react';

interface AuthModalProps {
  onClose: () => void;
  showModal: boolean;
}

const AuthModal: React.FC<AuthModalProps> = ({ onClose, showModal }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login, register, loading } = useAuth();

  if (!showModal) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      if (isLogin) {
        await login({ username, password });
      } else {
        await register({ username, password });
      }
      onClose(); // Close modal on successful auth
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-container p-8 rounded-lg shadow-lg max-w-sm w-full relative">
        <button onClick={onClose} className="absolute top-4 right-4 text-subtle hover:text-text">
          <X size={20} />
        </button>
        <h2 className="text-2xl font-bold text-primary mb-6 text-center">
          {isLogin ? 'Login' : 'Register'}
        </h2>
        
        {error && <p className="text-error text-center mb-4">{error}</p>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-subtle mb-1">Username</label>
            <input
              type="text"
              id="username"
              className="w-full px-3 py-2 bg-bg border border-gray-600 rounded-md text-text focus:outline-none focus:ring-2 focus:ring-primary"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-subtle mb-1">Password</label>
            <input
              type="password"
              id="password"
              className="w-full px-3 py-2 bg-bg border border-gray-600 rounded-md text-text focus:outline-none focus:ring-2 focus:ring-primary"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-primary text-bg py-2 rounded-md font-semibold hover:opacity-90 transition disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? 'Processing...' : (isLogin ? 'Login' : 'Register')}
          </button>
        </form>

        <p className="text-center text-subtle text-sm mt-6">
          {isLogin ? "Don't have an account?" : "Already have an account?"}{' '}
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-primary hover:underline font-medium"
            disabled={loading}
          >
            {isLogin ? 'Register' : 'Login'}
          </button>
        </p>
      </div>
    </div>
  );
};

export default AuthModal;
