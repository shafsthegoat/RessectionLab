import assert from "node:assert/strict";
import { test } from "node:test";
import { createHash } from "node:crypto";
import { readOnlyPreview } from "../src/preview-api.ts";

const bytes = new Uint8Array([1, 2, 3, 4]);
const digest = createHash("sha256").update(bytes).digest("hex");
function manifest() {
  return {
    case: {
      mri: { assetId: digest, sha256: digest, byteLength: 4 },
      compartments: [],
    },
    assets: { [digest]: `${digest}.bin` },
  };
}
async function withFetch(value, data, check) {
  const original = globalThis.fetch;
  globalThis.fetch = async (path) =>
    path === "/preview/manifest.json"
      ? Response.json(value)
      : new Response(data);
  try {
    await check();
  } finally {
    globalThis.fetch = original;
  }
}
test("read-only preview checks exact downloaded source digest", async () => {
  await withFetch(manifest(), bytes, async () => {
    const preview = await readOnlyPreview();
    assert.equal(preview.api.readOnly, true);
    assert.deepEqual(await preview.api.readAsset(digest), bytes);
    await assert.rejects(preview.api.trainPatient({}), /read-only/);
    await assert.rejects(preview.api.saveCase({}), /read-only/);
  });
});
test("corrupt same-length public assets fail checksum validation", async () => {
  await withFetch(manifest(), new Uint8Array([4, 3, 2, 1]), async () => {
    const preview = await readOnlyPreview();
    await assert.rejects(preview.api.readAsset(digest), /checksum/);
  });
});
test("preview accepts only controlled content-named binary files", async () => {
  const value = manifest();
  value.assets[digest] = "%2e%2e/private.bin";
  await withFetch(value, bytes, async () => {
    const preview = await readOnlyPreview();
    await assert.rejects(preview.api.readAsset(digest), /Unknown preview/);
  });
});
