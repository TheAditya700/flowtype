export interface SnippetResponse {
  id: string;
  words: string[];
}

export interface UserCreate {
  username: string;
  password: string;
}

export interface Token {
  access_token: string;
  token_type?: string;
}

export interface UserResponse {
  id: string;
  username?: string;
  is_anonymous?: boolean;
  created_at?: string;
  best_wpms?: Record<string, number>;
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
  recentSnippetIds?: string[];
  keystroke_timestamps?: number[];
}

export interface Snippet {
  id: string;
  words: string[];
  expectedWpm: number;
}

export interface SnippetResult {
  snippet_id: string;
  wpm: number;
  accuracy: number;
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
  snippets: SnippetResult[];
  user_state: UserState;
  sessionMode?: "15" | "30" | "60" | "120" | "free";
  flowScore?: number;
}

export interface SnippetLog {
  snippet_id: string;
  started_at: string;
  completed_at: string;
  wpm: number;
  accuracy: number;
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
  mode: "15" | "30" | "60" | "120";
}

export interface AnalyticsRequest {
  keystrokeData: KeystrokeEvent[];
  wpm: number;
  accuracy: number;
}

export interface AnalyticsResponse {
  // Shape can be refined when analytics endpoint stabilizes
  [key: string]: unknown;
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

// Observability Types
export interface ObservabilityHeader {
  total_sessions: number;
  active_users: number;
  model_version: string;
  last_snapshot_time: string | null;
}

export interface LearningHealthPoint {
  t: string;
  mean_precision: number;
  mean_variance: number;
}

export interface LearningHealthResponse {
  timeframe: string;
  points: LearningHealthPoint[];
}

export interface AgentEffectivenessPoint {
  t: string;
  mean_reward: number;
  reward_variance: number;
  reward_std: number;
  count: number;
}

export interface AgentEffectivenessResponse {
  timeframe: string;
  points: AgentEffectivenessPoint[];
}

export interface PerformanceDeltaPoint {
  t: string;
  delta_accuracy: number;
  delta_smoothness: number;
  delta_effective_wpm: number;
  actual_accuracy: number;
  actual_consistency: number;
  actual_effective_wpm: number;
}

export interface PerformanceDeltasResponse {
  timeframe: string;
  points: PerformanceDeltaPoint[];
}

export interface UserSkill {
  user_feature_idx: number;
  snippet_feature_idx?: number; // Only present in individual weight mode
  importance: number;
  precision: number;
  variance?: number; // Present in new aggregated mode
  mean_weight?: number; // Present in new aggregated mode
  sign: "positive" | "negative" | "mixed";
  interaction_count?: number; // Only present in old aggregation mode
}

export interface UserSkillsResponse {
  skills: UserSkill[];
}

export interface LearningActivityPoint {
  t: string;
  mean_abs_delta_mean: number;
  fraction_weights_updated: number;
}

export interface LearningActivityResponse {
  timeframe: string;
  points: LearningActivityPoint[];
}
