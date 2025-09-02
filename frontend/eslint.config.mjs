import { dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const eslintConfig = [
  // No extends at all disables all rules from presets
  // No rules means nothing is checked
];

export default eslintConfig;
