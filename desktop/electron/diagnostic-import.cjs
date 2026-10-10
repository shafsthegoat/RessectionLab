'use strict';
const {diagnosticRequest}=require('./diagnostic-contract.cjs');
async function importDiagnosticLayer(args,{pick,request}){
  const checked=diagnosticRequest(args);
  const descriptorPath=await pick('Select the saved unreviewed diagnostic display-layer.json',['json']);
  if(!descriptorPath)return null;
  return request('importDiagnosticLayer',{...checked,descriptorPath});
}
module.exports={importDiagnosticLayer};
