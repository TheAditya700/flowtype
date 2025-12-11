import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { changeUsername } from '../api/client';

const ChangeUsernamePage: React.FC = () => {
  const navigate = useNavigate();
  const { user, setUser } = useAuth();
  const [newUsername, setNewUsername] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!newUsername.trim()) {
      setError('Username cannot be empty');
      return;
    }

    if (newUsername === user?.username) {
      setError('New username must be different from current');
      return;
    }

    setLoading(true);
    try {
      const result = await changeUsername(newUsername);
      setUser(result);
      setSuccess(true);
      setTimeout(() => navigate('/'), 1500);
    } catch (err: any) {
      setError(err.message || 'Failed to change username');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto p-8 flex flex-col gap-6 min-h-[60vh]">
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-2 text-subtle hover:text-text transition w-fit"
      >
        <ArrowLeft size={16} />
        Back
      </button>

      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-text">Change Username</h1>
        <p className="text-subtle text-sm">Update your account username</p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">Current Username</label>
          <input
            type="text"
            disabled
            value={user?.username || ''}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-text-subtle text-sm"
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">New Username</label>
          <input
            type="text"
            value={newUsername}
            onChange={(e) => setNewUsername(e.target.value)}
            placeholder="Enter new username"
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-text text-sm focus:border-primary outline-none transition"
            disabled={loading}
          />
        </div>

        {error && (
          <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 text-sm text-red-400">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {success && (
          <div className="flex items-center gap-2 bg-green-500/10 border border-green-500/30 rounded-lg px-3 py-2 text-sm text-green-400">
            Username updated! Redirecting...
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="bg-primary text-bg font-semibold px-4 py-2 rounded-lg hover:opacity-90 transition disabled:opacity-50"
        >
          {loading ? 'Updating...' : 'Update Username'}
        </button>
      </form>
    </div>
  );
};

export default ChangeUsernamePage;
