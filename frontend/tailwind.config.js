/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Semantic colors mapped to CSS variables
        bg: 'var(--bg-color)',
        text: {
          DEFAULT: 'var(--text-color)',
          subtle: 'var(--text-subtle)',
        },
        primary: 'var(--primary-color)',
        error: 'var(--error-color)',
        success: 'var(--success-color)',
        container: 'var(--container-bg)',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Roboto Mono', 'monospace'], // Nerd fonts
      }
    },
  },
  plugins: [],
}
