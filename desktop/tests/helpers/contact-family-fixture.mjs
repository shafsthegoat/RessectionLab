import fs from 'node:fs';import{createHash}from'node:crypto';
// Exact saved root-planned controller outputs, transported through AssetRegistry.
// No trained artifact is present in either fixture.
export const familyFixture=(method='SEARCH')=>JSON.parse(fs.readFileSync(new URL(`../fixtures/contact-family-${method}.json`,import.meta.url),'utf8'));
export const familyCatalog=()=>JSON.parse(fs.readFileSync(new URL('../fixtures/contact-family-catalog.json',import.meta.url),'utf8'));
export const familyRequest=f=>{const e=f.result.episode;return{fixture:e.fixture,layoutId:e.layoutId,goalId:e.publicGoal.goalId,selector:e.selector}};
export const familyAssets=f=>({readAsset:async id=>new Uint8Array(Buffer.from(f.assets[id],'base64'))});
export function resealFamily(result){const{episodeId,...body}=result.episode;result.episodeCanonicalJson=JSON.stringify(body);result.episode.episodeId='sha256:'+createHash('sha256').update(result.episodeCanonicalJson).digest('hex');}
