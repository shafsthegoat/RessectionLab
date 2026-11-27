'use strict';
// Cold admission verifies the saved source and complete action history by
// replay. Its budget is a stall window: every progress event from the engine
// (one per replayed action, at most every half second) renews it, so a long but
// advancing verification keeps running and stays cancellable, while an engine
// that stops reporting still fails. The window equals the former fixed hour, so
// no admission that used to finish can now time out; a day-long ceiling bounds
// the whole request. Ordinary admitted instrument actions keep fixed budgets.
const ADMISSION_COMMANDS=new Set(['inspect','seek','seekFrame','selectBranch','undo','redo','selectScene']);
const ADMISSION_STALL_MS=3600000;
const ADMISSION_CEILING_MS=86400000;
const admission=(operation,args)=>operation==='surgicalRehearsal'&&ADMISSION_COMMANDS.has(args?.command);
const surgicalTimeoutMs=command=>ADMISSION_COMMANDS.has(command)?ADMISSION_STALL_MS:command==='generatePreparations'?600000:['fullCase','plan'].includes(command)?300000:120000;
const surgicalTransportLimitMs=(operation,args)=>operation==='anatomyReconstruct'?1800000:operation==='anatomyPrepare'&&args?.action==='run'?3600000:admission(operation,args)?3600000:operation==='surgicalRehearsal'&&args?.command==='generatePreparations'?600000:300000;
// Null for fixed budgets; otherwise the absolute ceiling of a progress-renewed request.
const progressRenewalCeilingMs=(operation,args)=>admission(operation,args)?ADMISSION_CEILING_MS:null;
module.exports={surgicalTimeoutMs,surgicalTransportLimitMs,progressRenewalCeilingMs,ADMISSION_STALL_MS,ADMISSION_CEILING_MS};
