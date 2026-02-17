/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: {
          bg: '#0a0f1a',
          grid: '#1a2035',
        },
        node: {
          dataset: '#10b981',
          transformer: '#3b82f6',
          estimator: '#8b5cf6',
          final: '#ef4444',
        },
      },
    },
  },
  plugins: [],
};
