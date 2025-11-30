import { UserState, TypingSession, SnippetResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export async function fetchNextSnippet(userState: UserState): Promise<SnippetResponse[]> {
  const response = await fetch(`${API_BASE}/snippets/retrieve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_state: userState })
  });
  
  if (!response.ok) {
    throw new Error('Failed to fetch snippet');
  }
  
  return response.json() as Promise<SnippetResponse[]>;
}

export async function saveSession(session: TypingSession): Promise<void> {
  await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(session)
  });
}
