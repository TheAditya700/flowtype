import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { changePassword } from '../api/client';

const ChangePasswordPage: React.FC = () => {
  const navigate = useNavigate();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!currentPassword || !newPassword || !confirmPassword) {
      setError('All fields are required');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }

    if (newPassword.length < 6) {
      setError('New password must be at least 6 characters');
      return;
    }

    if (newPassword === currentPassword) {
      setError('New password must be different from current');
      return;
    }

    setLoading(true);
    try {
      await changePassword(currentPassword, newPassword);
      setSuccess(true);
      setTimeout(() => navigate('/'), 1500);
    } catch (err: any) {
      setError(err.message || 'Failed to change password');
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
        <h1 className="text-3xl font-bold text-text">Change Password</h1>
        <p className="text-subtle text-sm">Update your account password</p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">Current Password</label>
          <input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            placeholder="Enter current password"
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-text text-sm focus:border-primary outline-none transition"
            disabled={loading}
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">New Password</label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            placeholder="Enter new password"
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-text text-sm focus:border-primary outline-none transition"
            disabled={loading}
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-text">Confirm Password</label>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm new password"
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
            Password updated! Redirecting...
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="bg-primary text-bg font-semibold px-4 py-2 rounded-lg hover:opacity-90 transition disabled:opacity-50"
        >
          {loading ? 'Updating...' : 'Update Password'}
        </button>
      </form>
    </div>
  );
};

export default ChangePasswordPage;
