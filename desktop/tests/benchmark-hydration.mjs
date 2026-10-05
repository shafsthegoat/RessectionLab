/** Warm-cache source hydration microbenchmark; not a GPU or UI responsiveness test. */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { hydrateCase, initialCursor } from "../src/case-data.ts";

const repository = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const preview = path.join(repository, "desktop/public/preview");
const manifest = JSON.parse(
  await fs.readFile(path.join(preview, "manifest.json"), "utf8"),
);
const api = {
  readAsset: async (id) =>
    new Uint8Array(await fs.readFile(path.join(preview, manifest.assets[id]))),
};
// Check recorded source bytes outside the timed renderer work. The desktop
// transport performs its own checksum verification before handing over arrays.
for (const descriptor of [
  manifest.case.mri,
  ...manifest.case.compartments.map((layer) => layer.array),
]) {
  const bytes = await api.readAsset(descriptor.assetId);
  if (
    bytes.length !== descriptor.byteLength ||
    createHash("sha256").update(bytes).digest("hex") !== descriptor.sha256
  )
    throw new Error("Benchmark source bytes disagree with recorded input.");
}
const samples = [];
for (let run = 0; run < 3; run++) {
  const start = performance.now();
  const source = await hydrateCase(manifest.case, api);
  const hydrated = performance.now();
  const cursor = initialCursor(source);
  const finished = performance.now();
  samples.push({
    run: run + 1,
    hydrateMilliseconds: hydrated - start,
    targetCursorMilliseconds: finished - hydrated,
    totalMilliseconds: finished - start,
    cursorRasMm: cursor,
  });
}
const result = {
  sourceChecksumsVerified: true,
  caseHash: manifest.case.caseHash,
  caseId: manifest.case.caseId,
  shape: manifest.case.shape,
  sourceBytes:
    manifest.case.mri.byteLength +
    manifest.case.compartments.reduce(
      (sum, layer) => sum + layer.array.byteLength,
      0,
    ),
  runtime: process.version,
  scope:
    "Local Node/V8 CPU and filesystem microbenchmark, three warm-cache sequential runs; not browser responsiveness or GPU latency.",
  samples,
  medianTotalMilliseconds: [...samples].sort(
    (a, b) => a.totalMilliseconds - b.totalMilliseconds,
  )[1].totalMilliseconds,
};
if (process.argv[2]) {
  const output = path.resolve(process.argv[2]);
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, JSON.stringify(result, null, 2) + "\n");
}
process.stdout.write(JSON.stringify(result, null, 2) + "\n");
