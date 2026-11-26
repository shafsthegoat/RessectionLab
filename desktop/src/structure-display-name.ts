/**
 * Short row wording for the structure inventory: a sentence-case name (and
 * the study-target role) on the first line and a muted detail
 * (quantity · provenance) on the second.
 *
 * The full source label stays the row's tooltip, accessible name and data
 * attribute. Wording never hides whether a structure is supplied, estimated,
 * simulated or generated: a provenance word removed from the name moves into
 * the detail.
 */
export type StructureProvenance='supplied'|'estimate'|'simulated'|'generated'|'unknown';

export interface StructureDisplayInput {
  /** Original source label. */
  label:string;
  /** Layer kind, e.g. `tumor`, `artery`, `brain-region`, `tract`. */
  kind:string;
  /** Study role; `target` marks the study's planning target. */
  sourceRole?:string;
  /** Layer provenance record, or a legacy provenance word such as `estimated`. */
  provenance?:unknown;
  /**
   * Source-cell volume in mm³: source voxel count × source voxel volume.
   * The sidecar omits it when the source grid has no physical units; then no
   * number is shown. Never pass a mesh or display-grid volume.
   */
  volumeMm3?:number|null;
  /** Source voxel count from anatomy QA. Zero marks an empty source mask. */
  voxelCount?:number|null;
}

export interface StructureDisplayName {
  /** Short sentence-case name. */
  name:string;
  /** Second line: quantity, provenance and caveats. */
  detail:string;
  /** The detail without its quantity, for rows that show a state instead. */
  qualifiers:string;
  /** Formatted source volume or `Empty source mask`, when known. */
  quantity?:string;
  provenance:StructureProvenance;
  /** `Study target` for the study's planning target, shown beside the name. */
  role?:string;
}

const CAUTION:StructureProvenance[]=['simulated','generated','estimate','supplied'];
const PROVENANCE_TEXT:Record<StructureProvenance,string>={supplied:'Supplied',estimate:'Estimate',simulated:'Simulated',generated:'Generated',unknown:''};
const ESTIMATE_KINDS=/estimate|inferred|prediction|proposal/;
const SUPPLIED_KINDS=/supplied|annotation|source-provided|observed/;
const MODEL_NAMES=/\b(?:SIAM|SuperSynth|SynthSeg|SynthStrip|FastSurfer|VesselFM|MONAI|BraTS|HD-BET)\b/i;
const PROPER_NOUNS=new Set(['broca','wernicke','sylvian','heschl','rolandic','willis','galen','labbé','labbe','trolard','meynert']);
// Short names for recurring source wordings, matched after provenance and
// dataset words are removed. Unlisted labels keep their own anatomy words.
const SYNONYMS:Array<[RegExp,string]>=[
  [/^non-?enhancing(?: tumou?r)?(?: core)?\s*(?:\/|or)\s*necro(?:tic|sis)(?: tumou?r)?(?: core)?$/i,'necrotic core'],
  [/^enhancing (?:target|lesion)$/i,'enhancing tumor'],
  [/^peritumou?ral (flair abnormality|edema|oedema)$/i,'$1'],
  [/^mineral(?:ized|ised) skull$/i,'skull'],
  [/^arterial$/i,'arteries'],
  [/^(cerebral )?venous$/i,'$1veins'],
];

function provenanceOfRecord(provenance:unknown):StructureProvenance|undefined {
  if(typeof provenance==='string')return provenanceOfWords(provenance);
  if(!provenance||typeof provenance!=='object')return undefined;
  const record=provenance as Record<string,unknown>;
  const kind=typeof record.kind==='string'?record.kind.toLowerCase():'';
  if(/simulat|synthetic/.test(kind))return 'simulated';
  if(/generated/.test(kind))return 'generated';
  if(ESTIMATE_KINDS.test(kind)||record.model!=null||record.inference!=null)return 'estimate';
  if(SUPPLIED_KINDS.test(kind))return 'supplied';
  return undefined;
}

function provenanceOfWords(text:string):StructureProvenance|undefined {
  const words=text.toLowerCase();
  if(/\b(?:simulated|synthetic)\b/.test(words))return 'simulated';
  if(/\bgenerated\b/.test(words))return 'generated';
  if(/\b(?:estimated?|inferred|approximate|proposal)\b/.test(words)||words==='image_derived_estimate'||words==='model_estimate')return 'estimate';
  if(/\b(?:supplied|source-provided)\b|^supplied_annotation$/.test(words))return 'supplied';
  return undefined;
}

/** The most cautious claim wins: never call an estimate supplied. */
function cautious(values:Array<StructureProvenance|undefined>):StructureProvenance {
  return CAUTION.find(value=>values.includes(value))??'unknown';
}

function sentenceCase(text:string):string {
  return text.split(' ').map((word,index)=>{
    if(index===0)return word.charAt(0).toUpperCase()+word.slice(1);
    return /^[A-Z][a-z]+(?:-[a-z]+)*$/.test(word)&&!PROPER_NOUNS.has(word.toLowerCase())?word.toLowerCase():word;
  }).join(' ');
}

interface LabelParts {name:string;hints:StructureProvenance[];sources:string[];caveats:string[]}

function labelParts(label:string):LabelParts {
  const hints:StructureProvenance[]=[],sources:string[]=[],caveats:string[]=[],kept:string[]=[];
  const [head='',...qualifiers]=label.replaceAll('_',' ').replace(/\s+/g,' ').trim().split(/\s+·\s+/);
  let base=head.replace(/\s*\(([^)]*)\)/g,(_,inner:string)=>{
    const words=inner.trim();
    if(/^estimated?$/i.test(words))hints.push('estimate');
    else if(words)caveats.push(sentenceCase(words.toLowerCase()));
    return '';
  }).trim();
  base=base.replace(/\bTopBrain(?:\s+v\d+)?\b/gi,' ').replace(/\s+/g,' ').trim();
  for(let match;(match=/^(supplied|estimated|simulated|generated|synthetic|source)\s+(?=\S)/i.exec(base));){
    const hint=provenanceOfWords(match[1]);
    if(hint)hints.push(hint);
    base=base.slice(match[0].length);
  }
  for(let match;(match=/\s+(contour|envelope|extraction|mask|labels?|segmentation|annotation|proposal|estimate)$/i.exec(base));){
    if(/proposal|estimate/i.test(match[1]))hints.push('estimate');
    base=base.slice(0,match.index);
  }
  const synonym=SYNONYMS.find(([pattern])=>pattern.test(base));
  if(synonym)base=base.replace(synonym[0],synonym[1]);
  for(const qualifier of qualifiers){
    if(/^(?:source )?label \d+$/i.test(qualifier))continue;
    if(/\b(?:partial|incomplete)\b/i.test(qualifier)){caveats.push('Partial');continue;}
    if(/^source-grid preview$/i.test(qualifier)){caveats.push('Source-grid preview');continue;}
    const hint=provenanceOfWords(qualifier);
    if(hint==='estimate'||MODEL_NAMES.test(qualifier)){
      hints.push('estimate');
      const source=qualifier.replace(/\s*\bestimate$/i,'').trim();
      if(source)sources.push(source);
      continue;
    }
    if(hint){hints.push(hint);continue;}
    kept.push(qualifier);
  }
  return {name:sentenceCase([base||head,...kept].join(' · ')),hints,sources,caveats};
}

/** A source-cell volume in cm³, or undefined when it cannot be stated. */
export function formatStructureVolume(volumeMm3:number|null|undefined):string|undefined {
  if(typeof volumeMm3!=='number'||!Number.isFinite(volumeMm3)||volumeMm3<=0)return undefined;
  const cm3=volumeMm3/1000;
  if(cm3<.01)return '<0.01 cm³';
  return (cm3>=100?cm3.toFixed(0):cm3>=1?cm3.toFixed(1):cm3.toFixed(2))+' cm³';
}

export function structureDisplayName(input:StructureDisplayInput):StructureDisplayName {
  const parts=labelParts(input.label);
  const recorded=provenanceOfRecord(input.provenance);
  // Tractography in this inventory is supplied bundle geometry.
  const provenance=cautious([recorded,...parts.hints,...(input.kind==='tract'&&!recorded&&!parts.hints.length?['supplied' as const]:[])]);
  const empty=input.voxelCount===0||input.volumeMm3===0;
  const consistent=!(typeof input.voxelCount==='number'&&input.voxelCount>0&&input.volumeMm3===0);
  const quantity=empty&&consistent?'Empty source mask':formatStructureVolume(input.voxelCount===0?undefined:input.volumeMm3);
  const source=provenance==='estimate'?parts.sources[0]:undefined;
  const qualifiers=[PROVENANCE_TEXT[provenance],source,...parts.caveats].filter((part):part is string=>!!part);
  return {
    name:parts.name,
    detail:[quantity,...qualifiers].filter(Boolean).join(' · '),
    qualifiers:qualifiers.join(' · '),
    ...(quantity?{quantity}:{}),
    provenance,
    ...(input.sourceRole==='target'?{role:'Study target'}:{}),
  };
}
