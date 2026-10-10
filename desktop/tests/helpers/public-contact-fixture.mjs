import fs from 'node:fs';import assert from 'node:assert/strict';import {createHash}from'node:crypto';
export function contactFixture(name='near-scripted'){
 const f=JSON.parse(fs.readFileSync(new URL('../fixtures/development-scripted.json',import.meta.url),'utf8')),contact=JSON.parse(fs.readFileSync(new URL(`../fixtures/public-contact-${name}.json`,import.meta.url),'utf8'));
 assert.equal(f.result.case.caseHash,contact.episode.caseHash);assert.deepEqual(f.result.episode.sourceBinding,contact.episode.sourceBinding);
 f.result={case:f.result.case,...contact};return f;
}
export const assetApi=f=>({readAsset:async id=>new Uint8Array(Buffer.from(f.assets[id],'base64'))});
export function reseal(result){const{episodeId,...body}=result.episode;result.episodeCanonicalJson=JSON.stringify(body);result.episode.episodeId='sha256:'+createHash('sha256').update(result.episodeCanonicalJson).digest('hex');}
