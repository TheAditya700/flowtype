export const generateResultCardImage = (
  wpm: number,
  accuracy: number,
  flowScore: number,
  canvas: HTMLCanvasElement
): Promise<string> => {
  return new Promise((resolve) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return resolve('');
    }

    const width = 800;
    const height = 400;
    canvas.width = width;
    canvas.height = height;

    // Background
    ctx.fillStyle = '#1a202c'; // Tailwind gray-900
    ctx.fillRect(0, 0, width, height);

    // Title
    ctx.font = 'bold 48px sans-serif';
    ctx.fillStyle = '#63b3ed'; // Tailwind blue-400
    ctx.textAlign = 'center';
    ctx.fillText('FlowType Session Result', width / 2, 80);

    // Stats
    ctx.font = 'normal 36px sans-serif';
    ctx.fillStyle = '#e2e8f0'; // Tailwind gray-200
    ctx.fillText(`WPM: ${Math.round(wpm)}`, width / 2, 180);
    ctx.fillText(`Accuracy: ${Math.round(accuracy)}%`, width / 2, 240);
    ctx.fillText(`FlowScore: ${flowScore.toFixed(1)}`, width / 2, 300);

    // Footer
    ctx.font = 'normal 20px sans-serif';
    ctx.fillStyle = '#a0aec0'; // Tailwind gray-500
    ctx.fillText('flowtype.app', width / 2, height - 30);

    resolve(canvas.toDataURL('image/png'));
  });
};
