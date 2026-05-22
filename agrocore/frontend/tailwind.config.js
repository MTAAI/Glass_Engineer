/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Vazirmatn", "Tahoma", "Arial", "sans-serif"],
      },
      colors: {
        primary: { DEFAULT: "#16a34a", 50: "#f0fdf4", 100: "#dcfce7", 500: "#22c55e", 600: "#16a34a", 700: "#15803d" },
        soil: "#92400e",
        water: "#0369a1",
        livestock: "#7c3aed",
        processing: "#ea580c",
        kitchen: "#be185d",
      },
    },
  },
  plugins: [],
};
