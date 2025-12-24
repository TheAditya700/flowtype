// Suppress noisy Recharts container size warnings in development
// without hiding unrelated console errors/warnings.

const shouldSuppress = (args: any[]): boolean => {
  if (!args || args.length === 0) return false;
  const msg = String(args[0] ?? "");
  // Match: The width(-1) and height(-1) of chart should be greater than 0
  return /The width\(-?\d+\) and height\(-?\d+\) of chart should be greater than 0/.test(
    msg,
  );
};

export function installRechartsConsoleFilter() {
  if (typeof window === "undefined") return;
  const isDev = (import.meta as any).env?.MODE !== "production";
  if (!isDev) return;

  const originalWarn = console.warn.bind(console);
  const originalError = console.error.bind(console);

  console.warn = (...args: any[]) => {
    if (shouldSuppress(args)) return;
    originalWarn(...args);
  };

  console.error = (...args: any[]) => {
    if (shouldSuppress(args)) return;
    originalError(...args);
  };
}
