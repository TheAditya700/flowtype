import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import App from './App.tsx';
import StatsPage from './pages/StatsPage.tsx';
import WikiPage from './pages/WikiPage.tsx';
import AuthPage from './pages/AuthPage.tsx';
import LeaderboardPage from './pages/LeaderboardPage.tsx';
import ChangeUsernamePage from './pages/ChangeUsernamePage.tsx';
import ChangePasswordPage from './pages/ChangePasswordPage.tsx';
import DeleteAccountPage from './pages/DeleteAccountPage.tsx';
import './index.css'; // Tailwind CSS import
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider } from './context/AuthContext'; // Import AuthProvider
import { SessionModeProvider, useSessionMode } from './context/SessionModeContext';
import Header from './components/Header';

function Root() {
  const location = useLocation();
  const { isPaused, setIsPaused, sessionStarted, setSessionStarted } = useSessionMode();

  // Reset header state when navigating away from type page
  useEffect(() => {
    if (location.pathname !== '/') {
      setIsPaused(true);
      setSessionStarted(false);
    }
  }, [location.pathname, setIsPaused, setSessionStarted]);

  return (
    <div className="min-h-screen bg-bg text-text flex flex-col items-center justify-start p-4 sm:p-8">
      <Header isPaused={isPaused} sessionStarted={sessionStarted} />
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/stats" element={<StatsPage />} />
        <Route path="/leaderboard" element={<LeaderboardPage />} />
        <Route path="/wiki" element={<WikiPage />} />
        <Route path="/auth" element={<AuthPage />} />
        <Route path="/account/change-username" element={<ChangeUsernamePage />} />
        <Route path="/account/change-password" element={<ChangePasswordPage />} />
        <Route path="/account/delete" element={<DeleteAccountPage />} />
      </Routes>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider> {/* Wrap with AuthProvider */}
      <SessionModeProvider>
        <ThemeProvider>
          <BrowserRouter>
            <Root />
          </BrowserRouter>
        </ThemeProvider>
      </SessionModeProvider>
    </AuthProvider>
  </React.StrictMode>,
);
