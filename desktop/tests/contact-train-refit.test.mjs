// Handwritten metadata boundary controls, not live trained/native execution.
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createRequire} from 'node:module';
import {checkedFamilyAvailability, requireInteractiveFamilyRequest} from '../src/contact-family-availability.ts';
import {checkedFamilyExecution} from '../src/contact-family-data.ts';
import {unavailableCatalog} from './helpers/contact-family-catalog.mjs';
const require=createRequire(import.meta.url), host=require('../electron/contact-family-authority.cjs');
const h='a'.repeat(64), sha='sha256:'+h;
function catalog(){const c=unavailableCatalog();c.version='generated-public-contact-learning-availability-v2';
  c.methods.IL_TRAIN_REFIT={available:true,reason:null,allowedRoles:['TRAIN'],
    experimentHash:'sha256:fd211e00dbe6dd1c9ed50a58d6c3b9f9f8896a0acd3a7815b2c52e4a8dc1014f',
    releaseHash:'sha256:68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887',
    checkpointFileSha256:'5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7',
    parameterHash:'sha256:0bdd6713358937ac2c665ff700210b24eb6a88433f6f410fc87bb62f23f7fe20',
    evidence:{fitResultSha256:h,rolloutResultSha256:h,independentAuditSha256:h},
    trainingBudget:{updates:32,statesPerUpdate:40,lossForwards:1280,fixedReadoutForwards:80},
    knownTRAINOutcome:{tasks:24,goalContacts:6,savedSEARCHContacts:16,STOPOnly:18,meanReturn:0.05633333333333332,scope:'generated_TRAIN_native_results_no_heldout_claim'}};return c;}
const request=()=>({fixture:'generated-public-contact-family-v2',layoutId:'pcf-00',goalId:'surface',selector:'IL_TRAIN_REFIT'});
function envelope(c){const m=c.methods.IL_TRAIN_REFIT;
  const author={experimentHash:m.experimentHash,architectureHash:sha,parameterHash:m.parameterHash,
    trainingLineageHash:sha,checkpointFileSha256:m.checkpointFileSha256};
  return {case:{},episode:{selector:'IL',layoutId:'pcf-00',publicGoal:{goalId:'surface'},splitRole:'TRAIN',familyHash:c.familyHash,learnedAuthorship:author},
    episodeCanonicalJson:'{}',policyVariant:'IL_TRAIN_REFIT',executionProvenance:{version:'generated-contact-train-refit-execution-v1',
      variant:'IL_TRAIN_REFIT',algorithm:'IL',layoutId:'pcf-00',goalId:'surface',splitRole:'TRAIN',familyHash:c.familyHash,
      ...author,releaseManifestSha256:m.releaseHash.slice(7),...m.evidence,completedUpdates:32,statesPerUpdate:40,
      inferenceOptimizerUpdates:0,ownedResultSha256:h,ownedSupervisionSha256:h}};}
test('v1 catalog and original negative slots remain exact; distinct v2 admits TRAIN only',()=>{
  assert.deepEqual(checkedFamilyAvailability(unavailableCatalog()),unavailableCatalog());
  const c=catalog();assert.equal(checkedFamilyAvailability(c).methods.IL.available,false);
  for(const api of [{requireInteractiveFamilyRequest},host]){
    assert.equal(api.requireInteractiveFamilyRequest(request(),c).layout.role,'TRAIN');
    for(const layoutId of ['pcf-12','pcf-20'])assert.throws(()=>api.requireInteractiveFamilyRequest({...request(),layoutId},c));
    assert.throws(()=>api.requireInteractiveFamilyRequest(request(),unavailableCatalog()));
  }
});
test('exact extra compute and all negative outcomes are required in both validators',()=>{
  for(const mutate of [c=>c.methods.IL_TRAIN_REFIT.knownTRAINOutcome.goalContacts=24,
    c=>c.methods.IL_TRAIN_REFIT.trainingBudget.lossForwards=128,
    c=>c.methods.IL_TRAIN_REFIT.allowedRoles.push('SELECT'),
    c=>c.methods.IL_TRAIN_REFIT.parameterHash=sha]){
    const c=catalog();mutate(c);assert.throws(()=>checkedFamilyAvailability(c));assert.throws(()=>host.checkedFamilyAvailability(c));
  }
});
test('distinct live provenance is internally checked and cannot replace original IL',()=>{
  const c=catalog(), r=envelope(c);
  for(const check of [checkedFamilyExecution,host.checkedFamilyExecution]){
    check(r,c,request());assert.throws(()=>check(r,c,{...request(),selector:'IL'}));
    for(const mutate of [r=>delete r.policyVariant,r=>r.executionProvenance.statesPerUpdate=4,
      r=>r.executionProvenance.rolloutResultSha256='b'.repeat(64),r=>r.episode.splitRole='SELECT',
      r=>r.executionProvenance.version='generated-contact-family-execution-v1',r=>r.executionProvenance.pilotResultSha256=h]){
      const bad=structuredClone(r);mutate(bad);assert.throws(()=>check(bad,c,request()));
    }
  }
});
