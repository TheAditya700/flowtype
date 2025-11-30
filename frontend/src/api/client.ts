import { UserState, TypingSession, SnippetResponse } from '../types';

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
