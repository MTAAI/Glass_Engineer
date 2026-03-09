/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        glass: {
          bg: '#0f1117',
          surface: '#1a1f2e',
          border: '#2d3748',
          blue: '#1e3a5f',
          accent: '#3b82f6',
          cyan: '#06b6d4',
          text: '#e2e8f0',
          muted: '#94a3b8',
          dim: '#64748b',
        },
      },
    },
  },
  plugins: [],
}
