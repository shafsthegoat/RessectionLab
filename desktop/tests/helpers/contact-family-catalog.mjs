// Handcrafted metadata transport control; no artifact, trained result or claim
// about the canonical role assignment is produced by this fixture.
export function unavailableCatalog(){return {
 version:'generated-public-contact-learning-availability-v1',fixture:'generated-public-contact-family-v2',
 familyHash:'sha256:'+'a'.repeat(64),experimentHash:null,releaseHash:null,
 layouts:Array.from({length:24},(_,i)=>({layoutId:`pcf-${String(i).padStart(2,'0')}`,role:i<12?'TRAIN':i<16?'SELECT':'MEASUREMENT_EVAL',goals:['surface','deep'],interactive:i<16})),
 methods:{STOP:{available:true,reason:null},SEARCH:{available:true,reason:null},IL:{available:false,reason:'final_artifacts_unavailable'},RL:{available:false,reason:'final_artifacts_unavailable'}}
};}
