/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/templates/**/*.html"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        dark: {
          900: "#0f1117",
          800: "#161925",
          700: "#1e2235",
          600: "#272b40",
          500: "#333855",
        },
        accent: {
          blue: "#3b82f6",
          green: "#22c55e",
          red: "#ef4444",
          yellow: "#eab308",
          purple: "#a855f7",
        },
      },
    },
  },
  plugins: [],
};
