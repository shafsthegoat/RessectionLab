/** Short statement of what an operative field draws, from each layer's
 * declared provenance. Estimates, assumptions and generated content stay
 * named separately; nothing undeclared is called acquired. */
type Layer = {kind?: string; displayKind?: string; provenance?: string};

const KIND_WORDS: Record<string, string> = {
  scalp: 'scalp', galea: 'galea', tissue: 'head tissue', extracranial: 'head tissue', skull: 'bone', diploe: 'bone',
  bone: 'bone', dura: 'dura', membrane: 'dura', grayMatter: 'cortex', whiteMatter: 'white matter', tract: 'tracts',
  tumor: 'tumour', artery: 'vessels', vein: 'vessels', vessel: 'vessels',
};
const GENERATED = new Set(['simulated', 'generated', 'synthetic']);

function group(provenance: string) {
  if (provenance === 'assumed') return 'assumed';
  if (GENERATED.has(provenance)) return 'generated';
  if (provenance === 'estimated' || provenance === 'model_estimate') return 'estimated';
  return provenance.replaceAll('_', ' ');
}

export function operativeAnatomyTitle(layers: readonly Layer[]): string {
  const declared = layers.filter((layer) => typeof layer.provenance === 'string' && layer.provenance);
  // Earlier generated scenes declare no layer provenance.
  if (!declared.length) return 'Generated anatomy';
  const groups = new Map<string, string[]>();
  for (const layer of declared) {
    const key = group(layer.provenance!),
      word = KIND_WORDS[layer.displayKind ?? layer.kind ?? ''] ?? 'tissue';
    const words = groups.get(key) ?? [];
    if (!words.includes(word)) words.push(word);
    groups.set(key, words);
  }
  if (groups.size === 1 && groups.has('generated') && declared.length === layers.length) return 'Generated anatomy';
  const parts: string[] = [];
  if (groups.has('estimated')) parts.push('Estimated anatomy');
  for (const [key, words] of groups) if (key !== 'estimated') parts.push(`${key} ${words.join(', ')}`);
  if (declared.length < layers.length) parts.push('other layers undeclared');
  const text = parts.join(' · ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Inventory text for one engine region, led by its declared provenance and review. */
export function regionProvenance(record: {provenance?: unknown; review_status?: unknown} | undefined, material: string, target: boolean): string {
  const base = `${material.replaceAll('_', ' ')} · ${target ? 'declared target' : 'context anatomy'}`;
  const declared = record?.provenance, review = record?.review_status;
  if (typeof declared !== 'string' || !declared) return base;
  const word = declared === 'assumed' ? 'Assumed layer' : GENERATED.has(declared) ? 'Generated'
    : declared === 'estimated' ? 'Model estimate' : declared.replaceAll('_', ' ');
  const reviewed = typeof review === 'string' && review && review !== declared ? `, ${review.replaceAll('_', ' ')}` : '';
  return `${word}${reviewed} · ${base}`;
}
