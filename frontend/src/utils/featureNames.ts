/**
 * Feature index to human-readable name mapping
 * User state is 130-dimensional:
 *   - Indices 0-56: EMA (Exponential Moving Average) of base features
 *   - Indices 57-113: STD (Standard Deviation) of base features
 *   - Indices 114-129: Previous snippet embedding (16-dim PCA)
 */

const BASE_FEATURE_NAMES: Record<number, string> = {
  // Accuracy (9 features, indices 0-8)
  0: "Accuracy",
  1: "Error Rate",
  2: "KSPC (Keystrokes Per Char)",
  3: "Backspace Ratio",
  4: "Burst Error Rate",
  5: "Accuracy (Left-to-Left)",
  6: "Accuracy (Right-to-Right)",
  7: "Accuracy (Cross-hand)",
  8: "Accuracy (Repeat key)",
  // Timing (12 features, indices 9-20)
  9: "IKI Mean (ms)",
  10: "IKI Std Dev",
  11: "IKI Coefficient of Variation",
  12: "IKI Mean (L2L)",
  13: "IKI CV (L2L)",
  14: "IKI Mean (R2R)",
  15: "IKI CV (R2R)",
  16: "IKI Mean (Cross-hand)",
  17: "IKI CV (Cross-hand)",
  18: "IKI Mean (Repeat)",
  19: "IKI CV (Repeat)",
  20: "Boundary Penalty",
  // Speed (2 features, indices 21-22)
  21: "WPM (Raw)",
  22: "WPM (Effective)",
  // Rollover (5 features, indices 23-27)
  23: "Rollover Rate",
  24: "Rollover Depth (ms)",
  25: "Rollover Rate (L2L)",
  26: "Rollover Rate (R2R)",
  27: "Rollover Rate (Cross-hand)",
  // Chunking (3 features, indices 28-30)
  28: "Spike Rate",
  29: "Flow Ratio",
  30: "Avg Chars per Chunk",
  // Letter Confidence (26 features, indices 31-56)
  31: "Confidence [a]",
  32: "Confidence [b]",
  33: "Confidence [c]",
  34: "Confidence [d]",
  35: "Confidence [e]",
  36: "Confidence [f]",
  37: "Confidence [g]",
  38: "Confidence [h]",
  39: "Confidence [i]",
  40: "Confidence [j]",
  41: "Confidence [k]",
  42: "Confidence [l]",
  43: "Confidence [m]",
  44: "Confidence [n]",
  45: "Confidence [o]",
  46: "Confidence [p]",
  47: "Confidence [q]",
  48: "Confidence [r]",
  49: "Confidence [s]",
  50: "Confidence [t]",
  51: "Confidence [u]",
  52: "Confidence [v]",
  53: "Confidence [w]",
  54: "Confidence [x]",
  55: "Confidence [y]",
  56: "Confidence [z]",
};

const USER_FEATURE_NAMES: Record<number, string> = {};

// Add EMA features (indices 0-56)
for (let i = 0; i < 57; i++) {
  USER_FEATURE_NAMES[i] = (BASE_FEATURE_NAMES[i] || `Feature ${i}`) + " (EMA)";
}

// Add STD features (indices 57-113)
for (let i = 57; i < 114; i++) {
  const baseIdx = i - 57;
  USER_FEATURE_NAMES[i] =
    (BASE_FEATURE_NAMES[baseIdx] || `Feature ${baseIdx}`) + " (Std Dev)";
}

// Add previous snippet embedding features (indices 114-129)
for (let i = 114; i < 130; i++) {
  USER_FEATURE_NAMES[i] = `Prev Snippet PCA ${i - 114 + 1}`;
}

const SNIPPET_FEATURE_NAMES: Record<number, string> = {
  0: "PCA Component 1",
  1: "PCA Component 2",
  2: "PCA Component 3",
  3: "PCA Component 4",
  4: "PCA Component 5",
  5: "PCA Component 6",
  6: "PCA Component 7",
  7: "PCA Component 8",
  8: "PCA Component 9",
  9: "PCA Component 10",
  10: "PCA Component 11",
  11: "PCA Component 12",
  12: "PCA Component 13",
  13: "PCA Component 14",
  14: "PCA Component 15",
  15: "PCA Component 16",
};

export function getFeatureName(
  featureIdx: number,
  type: "user" | "snippet" = "user",
): string {
  if (type === "user") {
    return USER_FEATURE_NAMES[featureIdx] || `User Feature ${featureIdx}`;
  } else {
    return SNIPPET_FEATURE_NAMES[featureIdx] || `Snippet Feature ${featureIdx}`;
  }
}

export function getInteractionName(
  snippetIdx: number,
  userIdx: number,
): string {
  const snippetName = getFeatureName(snippetIdx, "snippet");
  const userName = getFeatureName(userIdx, "user");
  return `${snippetName} ← → ${userName}`;
}
