import { UserState, TypingSession, SnippetResponse } from '../types';

// @ts-ignore - Vite env type
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export async function fetchNextSnippet(userState: UserState, currentSnippetId?: string): Promise<SnippetResponse | null> {
  const response = await fetch(`${API_BASE}/snippets/retrieve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_state: userState, current_snippet_id: currentSnippetId })
  });
  
  if (!response.ok) {
    throw new Error('Failed to fetch snippet');
  }
  
  // backend returns { snippet, wpm_windows }
  const data = await response.json();
  if (data && data.snippet) {
    return data.snippet as SnippetResponse;
  }
  // no snippet found
  return null;
}

export async function saveSession(session: TypingSession): Promise<void> {
  await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(session)
  });
}

export async function sendSnippetTelemetry(snippetLog: any, userState?: UserState): Promise<void> {
  // send snippet-level telemetry to backend for online tuning and offline training
  const payload = { snippet: snippetLog, user_state: userState };
  try {
    const res = await fetch(`${API_BASE}/telemetry/snippet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      console.warn('Telemetry post failed', res.status);
    }
  } catch (err) {
    console.warn('Failed to send telemetry', err);
  }
}
