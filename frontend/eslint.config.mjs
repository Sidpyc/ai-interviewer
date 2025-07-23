// frontend/eslint.config.js
import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends(
    "next/core-web-vitals",
    "next/typescript"
  ),
  // NEW: Add a configuration object specifically for custom rules
  {
    rules: {
      'react/no-unescaped-entities': 'off', // Disables the unescaped entities error
      '@next/next/no-page-custom-font': 'off', // Disables the custom font warning/error
    },
  },
];

export default eslintConfig;