import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: "#0a0a0a",
      },
    },
  },
  plugins: [],
};

export default config;
