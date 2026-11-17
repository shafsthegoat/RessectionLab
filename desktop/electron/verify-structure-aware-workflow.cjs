'use strict';
// Install fail-closed display guards before loading the real workflow harness.
const {app,BrowserWindow,dialog}=require('electron');
let failed=false;
const fail=error=>{if(!failed){failed=true;console.error('HIDDEN_PROCEDURE_FATAL '+String(error));}try{app.exit(1);}finally{process.exit(1);}};
process.prependListener('uncaughtException',fail);
process.prependListener('unhandledRejection',fail);
for(const method of ['show','showInactive','focus'])BrowserWindow.prototype[method]=function(){fail('FORBIDDEN_DISPLAY_'+method);};
dialog.showErrorBox=(title,message)=>fail(title+': '+message);
dialog.showOpenDialog=async()=>{throw Error('Unexpected native open dialog');};
dialog.showMessageBox=async()=>{throw Error('Unexpected native message dialog');};
app.dock?.hide();
process.env.WORKFLOW_CRANIAL_GENERATE='1';
process.env.WORKFLOW_PROCEDURE='1';
require('./verify-surgical-workflow.cjs');
