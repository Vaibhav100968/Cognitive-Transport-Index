/**
 * Strip Finder quarantine from native .node binaries under node_modules.
 * Without this, macOS can block @tailwindcss/oxide and lightningcss with:
 * "library load disallowed by system policy"
 */
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

if (process.platform !== "darwin") process.exit(0);

const nodeModules = path.join(__dirname, "..", "node_modules");
if (!fs.existsSync(nodeModules)) process.exit(0);

try {
  execFileSync("xattr", ["-cr", nodeModules], { stdio: "inherit" });
} catch {
  // non-fatal (e.g. sandboxed CI)
}
