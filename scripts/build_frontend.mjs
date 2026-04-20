import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stripTypeScriptTypes } from "node:module";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const sourcePath = path.join(root, "src", "web", "app.ts");
const outputPath = path.join(root, "src", "web", "app.js");

const source = await readFile(sourcePath, "utf8");
const transformed = stripTypeScriptTypes(source, {
    mode: "transform",
    sourceUrl: sourcePath,
});

const banner = "// Generated from src/web/app.ts by scripts/build_frontend.mjs\n";
await writeFile(outputPath, `${banner}${transformed}`, "utf8");

console.log(`Built ${path.relative(root, outputPath)}`);
