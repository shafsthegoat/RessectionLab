import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createRequire} from 'node:module';
import {checkedFamilyAvailability, requireInteractiveFamilyRequest} from '../src/contact-family-availability.ts';
import {checkedContactFamilyRequest} from '../src/contact-family-request.ts';
import {hydrateContactFamilyEpisode} from '../src/contact-family-data.ts';
import {contactFixture} from './helpers/public-contact-fixture.mjs';
const require=createRequire(import.meta.url),host=require('../electron/contact-family.cjs'),{Sidecar}=require('../electron/sidecar.cjs');
import {unavailableCatalog} from './helpers/contact-family-catalog.mjs';
const request=()=>({fixture:'generated-public-contact-family-v2',layoutId:'pcf-00',goalId:'surface',selector:'SEARCH'});
test('unreleased catalog is detached and explicitly disables both learned methods',()=>{const input=unavailableCatalog(),result=checkedFamilyAvailability(input);input.methods.IL.available=true;assert.equal(result.methods.IL.available,false);assert.equal(result.experimentHash,null);assert.equal(host.checkedFamilyAvailability(unavailableCatalog()).methods.RL.available,false);});
test('TRAIN/SELECT selectable; held-out and absent learned releases fail before any task',()=>{for(const layoutId of ['pcf-00','pcf-12'])assert.equal(requireInteractiveFamilyRequest({...request(),layoutId},unavailableCatalog()).layout.interactive,true);for(const change of [{layoutId:'pcf-16'},{selector:'IL'},{selector:'RL'}]){assert.throws(()=>requireInteractiveFamilyRequest({...request(),...change},unavailableCatalog()));assert.throws(()=>host.familyRequest({...request(),...change},unavailableCatalog()));}});
test('request cannot choose artifact, role, coordinates, old fixture or unknown method',()=>{for(const change of [{checkpoint:'/private'}, {role:'TRAIN'},{nativeIndex:[1,2,3]},{fixture:'generated-public-surface-contact-v1'},{layoutId:'pcf-24'},{selector:'HYBRID'},{goalId:'near'}])assert.throws(()=>checkedContactFamilyRequest({...request(),...change}));});
for(const[name,mutate]of[
 ['array hash coercion',c=>c.familyHash=[c.familyHash]],['array layout coercion',c=>c.layouts[0].layoutId=[c.layouts[0].layoutId]],
 ['missing layout',c=>c.layouts.pop()],['duplicate layout',c=>c.layouts[1].layoutId=c.layouts[0].layoutId],
 ['heldout admission',c=>c.layouts[20].interactive=true],['released flag without artifacts',c=>{c.methods.IL={available:true,reason:null};c.methods.RL={available:true,reason:null}}],
 ['ambiguous unavailability',c=>c.methods.IL.reason=null],['unexpected metadata',c=>c.privateReference={assetId:'private'}]
])test(`catalog refuses ${name}`,()=>{const c=unavailableCatalog();mutate(c);assert.throws(()=>checkedFamilyAvailability(c));assert.throws(()=>host.checkedFamilyAvailability(c));});
test('old v2 cannot be admitted as family v3 before assets',async()=>{const f=contactFixture();await assert.rejects(hydrateContactFamilyEpisode(f.result,{readAsset:()=>assert.fail('no assets')},request(),unavailableCatalog()));});
test('sidecar requires prior validated availability and refuses unreleased inference before write',async()=>{const side={closed:false,pending:new Map(),child:{stdin:{write:()=>assert.fail('no child request')}}};await assert.rejects(Sidecar.prototype.request.call(side,'executePublicContactFamilyEpisode',request(),30));side.contactFamilyAvailability=unavailableCatalog();await assert.rejects(Sidecar.prototype.request.call(side,'executePublicContactFamilyEpisode',{...request(),selector:'IL'},30));});
test('availability preserves case assets and malformed metadata cannot be published',async()=>{let exposed=0;const side={pending:new Map([['id',{op:'publicContactFamilyAvailability',resolve(){},timeout:null}]]),assets:{clear:()=>assert.fail('no clear'),expose:async r=>{exposed++;return r}},emit(){}};await Sidecar.prototype.handle.call(side,{id:'id',event:'result',result:unavailableCatalog()});assert.equal(exposed,1);assert.equal(side.contactFamilyAvailability.methods.IL.available,false);const bad=unavailableCatalog();bad.layouts[0].hidden={path:'/private',dtype:'uint8',byteLength:1};side.pending.set('bad',{op:'publicContactFamilyAvailability'});await assert.rejects(Sidecar.prototype.handle.call(side,{id:'bad',event:'result',result:bad}));assert.equal(exposed,1);});
