import React, { createContext, useState, useEffect, useContext, ReactNode, useCallback } from 'react';
import { loginUser, registerUser, fetchCurrentUser, removeToken, getToken, setToken } from '../api/client';
import { UserCreate, UserResponse } from '../types';

interface AuthContextType {
  user: UserResponse | null;
  loading: boolean;
  login: (credentials: UserCreate) => Promise<void>;
  register: (credentials: UserCreate) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const isAuthenticated = !!user;

  const loadUser = useCallback(async () => {
    setLoading(true);
    try {
      const storedToken = getToken();
      if (storedToken) {
        const currentUser = await fetchCurrentUser();
        setUser(currentUser);
      } else {
        setUser(null);
      }
    } catch (error) {
      console.error("Failed to load user:", error);
      removeToken(); // Clear invalid token
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  const login = async (credentials: UserCreate) => {
    setLoading(true);
    try {
      const token = await loginUser(credentials);
      if (token) {
        await loadUser(); // Fetch user data after successful login
      }
    } finally {
      setLoading(false);
    }
  };

  const register = async (credentials: UserCreate) => {
    setLoading(true);
    try {
      const newUser = await registerUser(credentials);
      // After registration, directly log them in or ask them to login
      // For simplicity, let's auto-login after successful registration
      if (newUser) {
        await login({ username: credentials.username, password: credentials.password });
      }
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    removeToken();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
