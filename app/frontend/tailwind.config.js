/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: {
          bg: '#000000',
          grid: '#0d1117',
        },
        node: {
          dataset: '#10b981',
          transformer: '#22d3ee',
          estimator: '#8b5cf6',
          final: '#f97316',
        },
        neon: {
          cyan:    '#22d3ee',
          orange:  '#f97316',
          emerald: '#10b981',
          purple:  '#a78bfa',
          red:     '#f87171',
        },
        brand: {
          bg:      '#000000',
          surface: '#03060a',
          panel:   '#060d14',
          border:  '#0e1f2e',
        },
      },
      boxShadow: {
        'glow-cyan':     '0 0 15px rgba(34,211,238,0.35)',
        'glow-cyan-lg':  '0 0 30px rgba(34,211,238,0.5)',
        'glow-orange':   '0 0 15px rgba(249,115,22,0.35)',
        'glow-emerald':  '0 0 15px rgba(16,185,129,0.35)',
        'glow-purple':   '0 0 15px rgba(167,139,250,0.35)',
        'glow-red':      '0 0 15px rgba(248,113,113,0.35)',
        'inner-cyan':    'inset 0 0 20px rgba(34,211,238,0.05)',
      },
      backdropBlur: {
        glass: '12px',
      },
    },
  },
  plugins: [],
};
