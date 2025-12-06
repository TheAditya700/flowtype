export interface SnippetResponse {
  id: string;
  words: string[];
  difficulty: number;
}

export interface KeystrokeEvent {
  timestamp: number;
  key: string;
  isBackspace: boolean;
  isCorrect: boolean;
}

export interface UserState {
  user_id?: string;
  rollingWpm: number;
  rollingAccuracy: number;
  backspaceRate: number;
  hesitationCount: number;
  recentErrors: string[];
  currentDifficulty: number;
  recentSnippetIds?: string[];
  keystroke_timestamps?: number[];
}

export interface Snippet {
  id: string;
  words: string[];
  difficulty: number;
  expectedWpm: number;
}

export interface SnippetResult {
  snippet_id: string;
  wpm: number;
  accuracy: number;
  difficulty: number;
  started_at?: number;
  completed_at?: number;
}

export interface SessionCreateRequest {
  user_id?: string;
  durationSeconds: number;
  wordsTyped: number;
  keystrokeData: KeystrokeEvent[];
  wpm: number;
  accuracy: number;
  errors: number;
  difficultyLevel: number;
  snippets: SnippetResult[];
  user_state: UserState;
  flowScore?: number;
}

export interface SnippetLog {
  snippet_id: string;
  started_at: string;
  completed_at: string;
  wpm: number;
  accuracy: number;
  difficulty: number;
}
