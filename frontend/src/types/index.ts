export interface SnippetResponse {
  id: string;
  words: string;
  difficulty: number;
}

export interface KeystrokeEvent {
  timestamp: number;
  key: string;
  isBackspace: boolean;
  isCorrect: boolean;
}

export interface UserState {
  rollingWpm: number;
  rollingAccuracy: number;
  backspaceRate: number;
  hesitationCount: number;
  recentErrors: string[];
  currentDifficulty: number;
}

export interface Snippet {
  id: string;
  words: string[];
  difficulty: number;
  expectedWpm: number;
}

export interface TypingSession {
  startedAt: Date;
  durationSeconds: number;
  wordsTyped: number;
  errors: number;
  wpm: number;
  accuracy: number;
  difficultyLevel: number;
  keystrokeData: KeystrokeEvent[];
}
