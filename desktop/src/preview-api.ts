import type { CasePayload, ResectionApi, SearchResult } from "./types";

interface PreviewManifest {
  case: CasePayload;
  assets: Record<string, string>;
  search?: SearchResult;
}

export async function readOnlyPreview(): Promise<{
  api: ResectionApi;
  caseData: CasePayload;
  search?: SearchResult;
} | null> {
  const response = await fetch("/preview/manifest.json");
  if (!response.ok || !response.headers.get("content-type")?.includes("json"))
    return null;
  const manifest: PreviewManifest = await response.json();
  const descriptors = [
    manifest.case.mri,
    manifest.case.brainMask,
    ...manifest.case.compartments.flatMap((layer) => [
      layer.array,
      layer.sourceArray,
    ]),
  ].filter(Boolean);
  const unavailable = async (): Promise<never> => {
    throw new Error(
      "This read-only browser preview does not run desktop operations.",
    );
  };
  const api: ResectionApi = {
    readOnly: true,
    ping: async () => ({ mode: "read_only_preview" }),
    startupCase: async () => manifest.case,
    createSyntheticCase: unavailable,
    openCase: unavailable,
    importNifti: unavailable,
    saveCase: unavailable,
    generateRoutes: unavailable,
    trainPatient: unavailable,
    listRuns: unavailable,
    replayTraining: unavailable,
    exportCandidate: unavailable,
    inspectEvidence: async () => ({
      metadata: manifest.case.metadata,
      unknowns: manifest.case.unknowns,
    }),
    cancel: unavailable,
    onEvent: () => () => {},
    readAsset: async (assetId: string) => {
      const path = manifest.assets[assetId];
      if (!path || !/^[a-f0-9]{64}\.bin$/.test(path))
        throw new Error("Unknown preview asset.");
      const asset = await fetch(`/preview/${path.replace(/^\//, "")}`);
      if (!asset.ok) throw new Error("Preview source asset is unavailable.");
      const descriptor = descriptors.find(
        (value) => value?.assetId === assetId,
      );
      if (!descriptor)
        throw new Error("Preview asset has no source descriptor.");
      const buffer = await asset.arrayBuffer();
      if (buffer.byteLength !== descriptor.byteLength)
        throw new Error("Preview source transfer is incomplete.");
      const digest = [
        ...new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
      ]
        .map((value) => value.toString(16).padStart(2, "0"))
        .join("");
      if (digest !== descriptor.sha256)
        throw new Error(
          "Preview source checksum does not match the recorded asset.",
        );
      return new Uint8Array(buffer);
    },
  };
  return { api, caseData: manifest.case, search: manifest.search };
}
