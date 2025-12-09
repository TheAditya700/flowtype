import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css'; // Tailwind CSS import
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider } from './context/AuthContext'; // Import AuthProvider
import { SessionModeProvider } from './context/SessionModeContext';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider> {/* Wrap with AuthProvider */}
      <SessionModeProvider>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </SessionModeProvider>
    </AuthProvider>
  </React.StrictMode>,
);
