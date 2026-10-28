'use strict';
// Cold admission verifies the saved source and complete action history. Keep
// that bounded separately from an ordinary already-admitted instrument action.
const ADMISSION_COMMANDS=new Set(['inspect','seek','seekFrame','selectBranch','undo','redo','selectScene']);
const surgicalTimeoutMs=command=>ADMISSION_COMMANDS.has(command)?3600000:['fullCase','plan'].includes(command)?300000:120000;
const surgicalTransportLimitMs=(operation,args)=>operation==='anatomyReconstruct'?1800000:operation==='surgicalRehearsal'&&ADMISSION_COMMANDS.has(args?.command)?3600000:300000;
module.exports={surgicalTimeoutMs,surgicalTransportLimitMs};
