import { format, subHours, subDays, subWeeks, subMonths } from "date-fns";

export type Timeframe = "hour" | "day" | "week" | "month";

export const getXAxisDomain = (timeframe: Timeframe, endTime?: number): [number, number] => {
  const end = endTime ? new Date(endTime) : new Date();
  let start = new Date(end);

  switch (timeframe) {
    case "hour":
      start = subHours(end, 1);
      break;
    case "day":
      start = subDays(end, 1);
      break;
    case "week":
      start = subWeeks(end, 1);
      break;
    case "month":
      start = subMonths(end, 1);
      break;
  }

  return [start.getTime(), end.getTime()];
};

export const formatXAxisTick = (timestamp: number | string, timeframe: Timeframe): string => {
  const date = new Date(timestamp);
  switch (timeframe) {
    case "hour":
      return format(date, "HH:mm"); // 14:30
    case "day":
      return format(date, "HH:mm"); // 14:00
    case "week":
      return format(date, "EEE HH:mm"); // Mon 14:00
    case "month":
      return format(date, "MMM dd"); // Jan 01
    default:
      return format(date, "MM/dd");
  }
};

/**
 * Generate 8 fixed time labels/ticks based on timeframe, working backwards from current time
 */
export const generateFixedTicks = (timeframe: Timeframe, endTime?: number): number[] => {
  const now = endTime ? new Date(endTime) : new Date();
  const ticks: number[] = [];
  
  switch (timeframe) {
    case "hour":
      // 8 ticks at 7.5 minute intervals going back 1 hour
      for (let i = 0; i < 8; i++) {
        const time = subHours(now, (i * 7.5) / 60);
        ticks.unshift(time.getTime());
      }
      break;
    case "day":
      // 8 ticks at 3 hour intervals going back 24 hours
      for (let i = 0; i < 8; i++) {
        const time = subHours(now, i * 3);
        ticks.unshift(time.getTime());
      }
      break;
    case "week":
      // 8 ticks at 21 hour intervals going back 7 days
      for (let i = 0; i < 8; i++) {
        const time = subHours(now, i * 21);
        ticks.unshift(time.getTime());
      }
      break;
    case "month":
      // 8 ticks at ~3.75 day intervals going back 30 days
      for (let i = 0; i < 8; i++) {
        const time = subDays(now, i * 3.75);
        ticks.unshift(time.getTime());
      }
      break;
  }
  
  return ticks;
};
