export type Scale = "single" | "x10" | "x100" | "x1000";

export const SCALE_GROUP_SIZE: Record<Scale, number> = {
  single: 1,
  x10: 10,
  x100: 100,
  x1000: 1000,
};

export const SCALE_POINTS = 10;

export const getScaleLimit = (scale: Scale): number => SCALE_GROUP_SIZE[scale] * SCALE_POINTS;

export const buildSessionAxis = (length: number): { domain: [number, number]; ticks: number[] } => {
  const end = Math.max(length, 1);
  const start = Math.max(1, end - SCALE_POINTS + 1);
  const ticks = Array.from({ length: end - start + 1 }, (_, idx) => start + idx);
  return { domain: [start, end], ticks };
};

const sessionRange = (index: number, scale: Scale): [number, number] => {
  const groupSize = SCALE_GROUP_SIZE[scale];
  const start = (index - 1) * groupSize + 1;
  const end = index * groupSize;
  return [start, end];
};

export const formatSessionTick = (index: number, scale: Scale): string => {
  const [start, end] = sessionRange(index, scale);
  return start === end ? `#${start}` : `${start}-${end}`;
};

export const formatSessionLabel = (index: number, scale: Scale): string => {
  const [start, end] = sessionRange(index, scale);
  return start === end ? `Session ${start}` : `Sessions ${start}-${end}`;
};
