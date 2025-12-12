import { UserState, SessionCreateRequest, SnippetResponse, UserCreate, Token, UserResponse, SnippetRetrieveResponse, UserProfile, SessionResponse, AnalyticsRequest, AnalyticsResponse, UserStatsDetail, LeaderboardEntry } from '../types';

// @ts-ignore - Vite env type
const API_BASE = import.meta.env.VITE_API_URL || '/api';

const TOKEN_KEY = 'flowtype_access_token';

// Helper to get/set JWT
export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (token: string) => localStorage.setItem(TOKEN_KEY, token);
export const removeToken = () => localStorage.removeItem(TOKEN_KEY);

// Generic API caller with auth capabilities
async function callApi<T>(
  endpoint: string,
  method: string = 'GET',
  body?: object,
  requiresAuth: boolean = false
): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (requiresAuth) {
    const token = getToken();
    if (!token) {
      throw new Error('Authentication required.');
    }
    headers['Authorization'] = `Bearer ${token}`;
  }

  const config: RequestInit = {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  };

  const response = await fetch(`${API_BASE}${endpoint}`, config);

  if (response.status === 401) {
    // Optionally handle token expiration/invalidity, e.g., redirect to login
    removeToken();
    // throw new Error('Unauthorized'); // Or handle gracefully
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || 'API request failed');
  }
  
  // Handle 204 No Content for cases where the API explicitly returns nothing
  if (response.status === 204) {
    return {} as T; // Return empty object for no content, for a Promise<void> type
  }

  return response.json() as Promise<T>; // Ensure it returns the JSON directly
}

// Auth API Calls
export async function registerUser(user: UserCreate): Promise<UserResponse> {
  return callApi<UserResponse>('/auth/register', 'POST', user);
}

export async function loginUser(credentials: UserCreate): Promise<Token> {
  const formBody = new URLSearchParams();
  formBody.append('username', credentials.username);
  formBody.append('password', credentials.password);

  const response = await fetch(`${API_BASE}/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formBody.toString(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(errorData.detail || 'Login failed');
  }

  const token: Token = await response.json();
  setToken(token.access_token);
  return token;
}

export async function fetchCurrentUser(): Promise<UserResponse> {
  return callApi<UserResponse>('/auth/users/me', 'GET', undefined, true);
}

export async function mergeProfiles(anonUserId: string, authUserId: string): Promise<{ success: boolean }> {
  return callApi<{ success: boolean }>('/profile/merge', 'POST', {
    anon_user_id: anonUserId,
    auth_user_id: authUserId
  });
}


// Existing API Calls, updated to use callApi
export async function fetchNextSnippet(userState: UserState, currentSnippetId?: string): Promise<SnippetRetrieveResponse | null> {
  const response = await callApi<SnippetRetrieveResponse>(
    '/snippets/retrieve',
    'POST',
    { user_state: userState, current_snippet_id: currentSnippetId }
  );
  
  return response;
}

export async function saveSession(session: SessionCreateRequest): Promise<SessionResponse> {
  return callApi<SessionResponse>('/sessions', 'POST', session);
}

// New: Fetch User Profile (for dashboard)
export async function fetchUserProfile(): Promise<UserProfile> {
    return callApi<UserProfile>(`/users/me/profile`, 'GET', undefined, true);
}

export async function fetchUserStatsDetail(userId: string): Promise<UserStatsDetail> {
  return callApi<UserStatsDetail>(`/users/${userId}/stats/detail`, 'GET');
}

export async function fetchLeaderboard(mode: '15' | '30' | '60' | '120' = '60', excludeAnon: boolean = false): Promise<LeaderboardEntry[]> {
  return callApi<LeaderboardEntry[]>(`/users/leaderboard?mode=${mode}&exclude_anon=${excludeAnon}`, 'GET');
}

// Account Management API Calls
export async function changeUsername(newUsername: string): Promise<UserResponse> {
  return callApi<UserResponse>('/auth/users/change-username', 'PUT', { new_username: newUsername }, true);
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<{ message: string }> {
  return callApi<{ message: string }>('/auth/users/change-password', 'PUT', { current_password: currentPassword, new_password: newPassword }, true);
}

export async function deleteAccount(password: string): Promise<{ message: string }> {
  return callApi<{ message: string }>('/auth/users/delete-account', 'DELETE', { password }, true);
}
