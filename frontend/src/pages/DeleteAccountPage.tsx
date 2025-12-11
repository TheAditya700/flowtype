import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { deleteAccount } from '../api/client';
import { useAuth } from '../context/AuthContext';

const DeleteAccountPage: React.FC = () => {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [password, setPassword] = useState('');
  const [confirmText, setConfirmText] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);

  const handleDelete = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!password) {
      setError('Password is required');
      return;
    }

    if (confirmText !== 'delete my account') {
      setError('Please type "delete my account" to confirm');
      return;
    }

    setLoading(true);
    try {
      await deleteAccount(password);
      await logout();
      navigate('/');
    } catch (err: any) {
      setError(err.message || 'Failed to delete account');
    } finally {
      setLoading(false);
    }
  };

  if (!showConfirmation) {
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
          <h1 className="text-3xl font-bold text-text">Delete Account</h1>
          <p className="text-subtle text-sm">This action cannot be undone</p>
        </div>

        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-400">
          <p className="font-semibold mb-1">Warning:</p>
          <p>Deleting your account will permanently remove all your data, including sessions, stats, and leaderboard entries.</p>
        </div>

        <button
          onClick={() => setShowConfirmation(true)}
          className="bg-red-600 text-white font-semibold px-4 py-2 rounded-lg hover:bg-red-700 transition"
        >
          Delete My Account
        </button>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md mx-auto p-8 flex flex-col gap-6 min-h-[60vh]">
      <button
        onClick={() => setShowConfirmation(false)}
        className="flex items-center gap-2 text-subtle hover:text-text transition w-fit"
      >
        <ArrowLeft size={16} />
        Back
      </button>

      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-text">Confirm Deletion</h1>
        <p className="text-subtle text-sm">Enter your password to confirm</p>
      </div>

      <form onSubmit={handleDelete} className="flex flex-col gap-4">
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-400">
          This will permanently delete your account and all associated data.
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-text text-sm focus:border-primary outline-none transition"
            disabled={loading}
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">Confirm</label>
          <input
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder='Type "delete my account"'
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-text text-sm focus:border-primary outline-none transition"
            disabled={loading}
          />
          <p className="text-xs text-subtle">Case-sensitive</p>
        </div>

        {error && (
          <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 text-sm text-red-400">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || confirmText !== 'delete my account'}
          className="bg-red-600 text-white font-semibold px-4 py-2 rounded-lg hover:bg-red-700 transition disabled:opacity-50"
        >
          {loading ? 'Deleting...' : 'Delete Account'}
        </button>
      </form>
    </div>
  );
};

export default DeleteAccountPage;
