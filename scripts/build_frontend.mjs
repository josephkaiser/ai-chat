import { access, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const webRoot = path.join(root, "src", "web");

async function listTypeScriptSources(dir) {
    const entries = await readdir(dir, { withFileTypes: true });
    const sources = [];
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            sources.push(...await listTypeScriptSources(fullPath));
            continue;
        }
        if (entry.isFile() && entry.name.endsWith(".ts")) {
            sources.push(fullPath);
        }
    }
    return sources.sort();
}

function outputPathFor(sourcePath) {
    return sourcePath.replace(/\.ts$/, ".js");
}

async function transformWithNodeModule(source, sourcePath) {
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

async function transformWithTypeScript(source, sourcePath) {
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

async function checkedInBundlesExist(sourcePaths) {
    const checks = sourcePaths.map(async (sourcePath) => {
        try {
            await access(outputPathFor(sourcePath), fsConstants.F_OK);
            return true;
        } catch {
            return false;
        }
    });
    const results = await Promise.all(checks);
    return results.every(Boolean);
}

const sourcePaths = await listTypeScriptSources(webRoot);
const transformedSources = [];

for (const sourcePath of sourcePaths) {
    const source = await readFile(sourcePath, "utf8");
    const transformed = await transformWithNodeModule(source, sourcePath) ?? await transformWithTypeScript(source, sourcePath);
    transformedSources.push({ sourcePath, transformed });
}

if (transformedSources.some(({ transformed }) => typeof transformed !== "string")) {
    if (await checkedInBundlesExist(sourcePaths)) {
        console.warn(
            "No compatible TypeScript transformer found. Keeping the checked-in src/web/*.js bundles."
        );
        process.exit(0);
    }
    throw new Error(
        "Unable to build src/web/*.js. Use Node >= 22.6 or install the 'typescript' package locally."
    );
}

for (const { sourcePath, transformed } of transformedSources) {
    const outputPath = outputPathFor(sourcePath);
    await mkdir(path.dirname(outputPath), { recursive: true });
    const banner = `// Generated from ${path.relative(root, sourcePath)} by scripts/build_frontend.mjs\n`;
    await writeFile(outputPath, `${banner}${transformed}`, "utf8");
}

console.log(
    `Built ${transformedSources.length} frontend module${transformedSources.length === 1 ? "" : "s"} in ${path.relative(root, webRoot)}`
);
