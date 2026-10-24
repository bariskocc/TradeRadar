/** @type {import('tailwindcss').Config} */
// Tema (24.10, kullanici istegi "dark mode gozumu yoruyor"): renkler CSS degiskenine bagli,
// degerler app/static/css/input.css'te. Varsayilan ACIK tema; eski koyu degerler `html.theme-dark`
// altinda duruyor. Sablonlardaki sinif adlari (bg-dark-800, text-white, text-gray-400 ...) degismedi:
// "dark-*" artik yuzey katmani, "white" ana metin rengi demek.
const v = (name) => `rgb(var(--tr-${name}) / <alpha-value>)`;

module.exports = {
  content: ["./app/templates/**/*.html"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        white: v("fg"),
        dark: {
          900: v("d900"),
          800: v("d800"),
          700: v("d700"),
          600: v("d600"),
          500: v("d500"),
        },
        gray: {
          200: v("g200"),
          300: v("g300"),
          400: v("g400"),
          500: v("g500"),
          600: v("g600"),
          700: v("g700"),
        },
        green: { 300: v("green3"), 400: v("green4") },
        red: { 300: v("red3"), 400: v("red4") },
        amber: { 400: v("amber4") },
        yellow: { 400: v("yellow4") },
        sky: { 400: v("sky4") },
        emerald: { 400: v("emerald4") },
        orange: { 400: v("orange4") },
        blue: { 400: v("blue4") },
        purple: { 400: v("purple4") },
        accent: {
          blue: v("blue"),
          green: v("green"),
          red: v("red"),
          yellow: v("yellow"),
          purple: v("purple"),
        },
      },
    },
  },
  plugins: [],
};
