export interface SnippetResponse {
  id: string;
  words: string[];
  difficulty: number;
}

export interface SnippetRetrieveResponse {
  snippet: SnippetResponse;
  wpm_windows: Record<string, number>;
}

export interface KeystrokeEvent {
  id?: string;
  timestamp: number;
  keyup_timestamp?: number;
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

export interface UserStats {
  total_sessions: number;
  avg_wpm: number;
  avg_accuracy: number;
  total_time_typing: number;
  best_wpm_15: number;
  best_wpm_30: number;
  best_wpm_60: number;
  best_wpm_120: number;
}

export interface UserProfile {
  user_id: string;
  username?: string;
  features: Record<string, any>;
  stats: UserStats;
}

export interface AnalyticsRequest {
  keystrokeData: KeystrokeEvent[];
  wpm: number;
  accuracy: number;
}

export interface SpeedPoint {
  time: number;
  wpm: number;
  rawWpm: number;
  errors: number;
}

export interface ReplayEvent {
  char: string;
  iki: number;
  isChunkStart: boolean;
  isError: boolean;
  snippetIndex?: number;
  isRollover?: boolean;
}

export interface AnalyticsResponse {
  smoothness: number;
  rollover: number;
  leftFluency: number;
  rightFluency: number;
  crossFluency: number;
  speed: number;
  accuracy: number;
  avgIki: number;
  kspc: number;
  errors: number;
  heatmapData: Record<string, { accuracy: number; speed: number }>;
  speedSeries: SpeedPoint[];
  replayEvents: ReplayEvent[];
  avgChunkLength: number;
}