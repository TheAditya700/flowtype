import { UserState, SessionCreateRequest, SnippetResponse, UserCreate, Token, UserResponse, SnippetRetrieveResponse, UserProfile, AnalyticsRequest, AnalyticsResponse } from '../types';

// @ts-ignore - Vite env type
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

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
  
  // Handle 204 No Content for cases like saveSession
  if (response.status === 204) {
    return {} as T; // Return empty object for no content
  }

  return response.json();
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


// Existing API Calls, updated to use callApi
export async function fetchNextSnippet(userState: UserState, currentSnippetId?: string): Promise<SnippetResponse | null> {
  const response = await callApi<SnippetRetrieveResponse>(
    '/snippets/retrieve',
    'POST',
    { user_state: userState, current_snippet_id: currentSnippetId }
  );
  
  if (response && response.snippet) {
    return response.snippet;
  }
  return null;
}

export async function saveSession(session: SessionCreateRequest): Promise<void> {
  // saveSession might need user_id from auth for proper telemetry
  // For now, it doesn't require auth token, as sessions can be anonymous.
  // If we decide to link all sessions to authenticated users, this would change.
  await callApi<void>('/sessions', 'POST', session);
}

// New: Fetch User Profile (for dashboard)
export async function fetchUserProfile(): Promise<UserProfile> {
    return callApi<UserProfile>(`/users/me/profile`, 'GET', undefined, true);
}

// New: Calculate Session Metrics
export async function calculateSessionMetrics(data: AnalyticsRequest): Promise<AnalyticsResponse> {
    return callApi<AnalyticsResponse>('/analytics/calculate', 'POST', data);
}
