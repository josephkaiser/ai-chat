import { access, readFile, writeFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const sourcePath = path.join(root, "src", "web", "app.ts");
const outputPath = path.join(root, "src", "web", "app.js");

async function transformWithNodeModule(source) {
    try {
        const moduleNs = await import("node:module");
        if (typeof moduleNs.stripTypeScriptTypes !== "function") {
            return null;
        }
        return moduleNs.stripTypeScriptTypes(source, {
            mode: "transform",
            sourceUrl: sourcePath,
        });
    } catch {
        return null;
    }
}

async function transformWithTypeScript(source) {
    try {
        const moduleNs = await import("node:module");
        const require = moduleNs.createRequire(import.meta.url);
        const ts = require("typescript");
        if (typeof ts?.transpileModule !== "function") {
            return null;
        }
        const result = ts.transpileModule(source, {
            compilerOptions: {
                module: ts.ModuleKind.ES2022,
                target: ts.ScriptTarget.ES2022,
            },
            fileName: sourcePath,
        });
        return typeof result.outputText === "string" ? result.outputText : null;
    } catch {
        return null;
    }
}

async function checkedInBundleExists() {
    try {
        await access(outputPath, fsConstants.F_OK);
        return true;
    } catch {
        return false;
    }
}

const source = await readFile(sourcePath, "utf8");
const transformed = await transformWithNodeModule(source) ?? await transformWithTypeScript(source);

if (typeof transformed !== "string") {
    if (await checkedInBundleExists()) {
        console.warn(
            "No compatible TypeScript transformer found. Keeping the checked-in src/web/app.js bundle."
        );
        process.exit(0);
    }
    throw new Error(
        "Unable to build src/web/app.js. Use Node >= 22.6 or install the 'typescript' package locally."
    );
}

const banner = "// Generated from src/web/app.ts by scripts/build_frontend.mjs\n";
await writeFile(outputPath, `${banner}${transformed}`, "utf8");

console.log(`Built ${path.relative(root, outputPath)}`);
