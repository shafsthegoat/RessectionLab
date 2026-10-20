/** Navigation preference only; the engine still admits the saved source/state. */
export function hasSavedFullCaseBranch(artifacts:Record<string,unknown>|undefined) {
  const record=artifacts?.surgical_toolbox as {branchId?:unknown;branches?:Record<string,unknown>;fullCase?:Record<string,{schema?:unknown}>}|undefined;
  return typeof record?.branchId==='string'&&!!record.branches?.[record.branchId]&&
    record.fullCase?.[record.branchId]?.schema==='resectionlab.full-case-rehearsal.v1';
}
