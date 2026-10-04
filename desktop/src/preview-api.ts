import type {CasePayload, ResectionApi, SearchResult} from './types';

interface PreviewManifest {case: CasePayload; assets: Record<string, string>; search?: SearchResult}

export async function readOnlyPreview(): Promise<{api: ResectionApi; caseData: CasePayload; search?: SearchResult} | null> {
  const response = await fetch('/preview/manifest.json');
  if (!response.ok || !response.headers.get('content-type')?.includes('json')) return null;
  const manifest: PreviewManifest = await response.json();
  const unavailable = async (): Promise<never> => {throw new Error('This read-only browser preview does not run desktop operations.');};
  const api: ResectionApi = {
    readOnly: true,
    ping: async () => ({mode: 'read_only_preview'}),
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
    inspectEvidence: async () => ({metadata: manifest.case.metadata, unknowns: manifest.case.unknowns}),
    cancel: unavailable,
    onEvent: () => () => {},
    readAsset: async (assetId: string) => {
      const path = manifest.assets[assetId];
      if (!path || path.includes('..') || path.includes('://')) throw new Error('Unknown preview asset.');
      const asset = await fetch(`/preview/${path.replace(/^\//, '')}`);
      if (!asset.ok) throw new Error('Preview source asset is unavailable.');
      return new Uint8Array(await asset.arrayBuffer());
    },
  };
  return {api, caseData: manifest.case, search: manifest.search};
}
