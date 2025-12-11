export interface SnippetResponse {
  id: string;
  words: string[];
  difficulty: number;
}

export interface SnippetRetrieveResponse {
  snippet: SnippetResponse;
  wpm_windows: Record<string, number>;
  predicted_wpm?: number;
  predicted_accuracy?: number;
  predicted_consistency?: number;
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
  is_partial?: boolean;
  completed_words?: number;
  total_words?: number;
}

export interface SessionCreateRequest {
  user_id?: string;
  durationSeconds: number;
  wordsTyped: number;
  keystrokeData: KeystrokeEvent[];
  difficultyLevel: number;
  snippets: SnippetResult[];
  user_state: UserState;
  sessionMode?: '15' | '30' | '60' | '120' | 'free';
  flowScore?: number;
  predicted_wpm?: number;
  predicted_accuracy?: number;
  predicted_consistency?: number;
}

export interface SnippetLog {
  snippet_id: string;
  started_at: string;
  completed_at: string;
  wpm: number;
  accuracy: number;
  difficulty: number;
  isPartial?: boolean;
  completedWords?: number;
  totalWords?: number;
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

export interface SessionTimeseriesPoint {
  timestamp: number;
  wpm: number;
  accuracy: number;
  raw_wpm?: number;
  ema_wpm?: number;
  ema_dev?: number;
  ema_accuracy?: number;
}

export interface ActivityDay {
  date: string; // YYYY-MM-DD
  count: number;
}

export interface UserStatsDetail {
  summary: UserStats;
  timeseries: SessionTimeseriesPoint[];
  activity: ActivityDay[];
  current_streak: number;
  longest_streak: number;
  char_heatmap: Record<string, { accuracy: number; speed: number }>;
}

export interface UserProfile {
  user_id: string;
  username?: string;
  features: Record<string, any>;
  stats: UserStats;
}

export interface LeaderboardEntry {
  user_id: string;
  username?: string;
  best_wpm: number;
  mode: '15' | '30' | '60' | '120';
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

// Merged SessionResponse - combines session metadata with all analytics
export interface SessionResponse {
  // Session metadata
  session_id: string;
  reward: number;
  durationSeconds: number;
  
  // Basic stats
  wpm: number;
  rawWpm: number;
  accuracy: number;
  errors: number;
  
  // Flow metrics
  smoothness: number;
  rollover: number;
  leftFluency: number;
  rightFluency: number;
  crossFluency: number;
  
  // Hand-specific rollover rates
  rolloverL2L: number;
  rolloverR2R: number;
  rolloverCross: number;
  
  // Detailed stats
  avgIki: number;
  kspc: number;
  avgChunkLength: number;
  heatmapData: Record<string, { accuracy: number; speed: number }>;
  
  // Time Series and Replay
  speedSeries: SpeedPoint[];
  replayEvents: ReplayEvent[];
  
  // Snippet results
  snippets?: SnippetResult[];
}