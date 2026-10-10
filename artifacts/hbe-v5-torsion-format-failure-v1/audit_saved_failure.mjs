// Saved metadata/source audit only. Never opens the experimental ZIP or curves.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';

const root=process.cwd();
const out='build/hbe-v5-torsion-failure-independent-v1';
const attempt='build/hbe-v5-torsion-evaluation-v1/attempt-01';
const sha=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const need=(v,m)=>{if(!v)throw Error(m)};
const read=p=>{const s=fs.lstatSync(p);need(s.isFile()&&!s.isSymbolicLink()&&s.size<=2**20,'small regular file: '+p);return fs.readFileSync(p)};
const json=p=>JSON.parse(read(p));
const bind=p=>({path:p,bytes:fs.statSync(p).size,sha256:sha(read(p))});
const write=(p,x)=>fs.writeFileSync(p,JSON.stringify(x,null,2)+'\n',{flag:'wx'});
const names=['access.jsonl','intent-receipt.json','intent.json','publication.json','readout-console.txt','receipt.json','state.json','terminal.json'];
need(JSON.stringify(fs.readdirSync(attempt).sort())===JSON.stringify(names),'exact failed-output inventory');
const original=names.map(n=>bind(attempt+'/'+n));
const state=json(attempt+'/state.json'),terminal=json(attempt+'/terminal.json'),publication=json(attempt+'/publication.json');
const receipt=json(attempt+'/receipt.json'),intent=json(attempt+'/intent.json');
const events=read(attempt+'/access.jsonl').toString().trim().split('\n').map(x=>JSON.parse(x));
const consoleText=read(attempt+'/readout-console.txt').toString();
const release=json(terminal.release.path);
need(sha(read(terminal.release.path))===terminal.release.sha256&&terminal.release.sha256==='6c0d33eb9fe19f562e70c7bdbd9a7a01f8296dbba46f4d32bd06e02181ffc0b2','exact root release');
need(publication.accepted===false&&publication.physical_validation_pass===null&&publication.terminal_sha256===sha(read(attempt+'/terminal.json')),'negative publication binding');
for(const r of [publication,intent,receipt])need(JSON.stringify(r.release)===JSON.stringify(terminal.release),'common release');
need(state.status===terminal.status&&state.status==='failed_torsion_evaluation_attempt','negative states');
need(state.error.type==='ValueError'&&state.error.message==="could not convert string to float: 'angle'",'strict parser error');
need(consoleText.includes('mechanics_hbe_access.py\", line 583')&&consoleText.includes('mechanics_hbe_access.py\", line 433')&&consoleText.endsWith("ValueError: could not convert string to float: 'angle'\n"),'saved source traceback');
need(state.held_out_access_attempted===true&&state.held_out_member_reads===null&&state.held_out_responses_accessed===null,'partial exposure retained');
need(events.length===2&&events[0].phase==='upstream_freeze_admitted'&&events[1].phase==='held_out_attempt','one admitted attempted exposure, no completion');
for(let i=0;i<2;i++)need(events[i].sequence===i&&events[i].release_sha256===terminal.release.sha256,'ledger identity');
need(JSON.stringify(events[0].upstream)===JSON.stringify(release.upstream),'upstream identity');
const declaration=json(release.declaration.path);
need(sha(read(release.declaration.path))===release.declaration.sha256,'declaration');
const roles=json(declaration.roles.path);
need(sha(read(declaration.roles.path))===declaration.roles.sha256,'roles');
need(events.every(e=>e.protocol_sha256===declaration.protocol.sha256),'protocol binding');
need(JSON.stringify(events[1].members)===JSON.stringify(roles.held_out_validation.members.map(x=>x.path)),'intended member order');
need(Object.values(declaration.csv_schemas).every(s=>s.header===null&&s.delimiter===','&&s.coordinate_column===0&&s.response_column===1&&s.coordinate_unit==='rad'&&s.response_unit==='Nm'),'unchanged strict schema');
for(const k of ['fit_calls','native_calls','mesher_calls'])need(state[k]===0&&terminal[k]===0,'no '+k);
need(state.new_freeze_saved===false&&state.physical_validation_pass===null&&state.empirical_tolerance===null,'no new freeze/physical conclusion');
const stage=terminal.readout_stage,c=terminal.cleanup;
need(terminal.readout_calls_attempted===1&&stage.exit_code===1&&stage.kill_reason===null,'single failed readout');
need(c.contained&&c.direct_child_reaped&&!c.fallback_used&&c.exit_code===1&&c.errors.length===0&&c.remaining_members.length===0,'clean failed-child cleanup');
need(stage.elapsed_seconds<60&&terminal.elapsed_before_terminal_publication_seconds<70&&stage.peak_sampled_process_group_rss_bytes<=536870912,'resource envelope');
need(original.reduce((n,f)=>n+f.bytes,0)<16777216,'final output cap');
const sourceCommit=release.source_commit;
const sourceChecks=[];
for(const [p,h] of Object.entries(release.source_bindings)){
  need(sha(read(p))===h,'current source '+p);
  need(sha(execFileSync('/usr/bin/git',['cat-file','blob',sourceCommit+':'+p]))===h,'historical source '+p);
  sourceChecks.push({path:p,sha256:h});
}
need(sha(execFileSync('/usr/bin/git',['cat-file','blob',sourceCommit+':'+release.declaration.path]))===release.declaration.sha256,'historical declaration');
const review=json(release.independent_source_review.path);
need(sha(read(release.independent_source_review.path))===release.independent_source_review.sha256&&review.decision==='GO','source review');
const fixity=[];
// Explicit original-freeze fixity request: digest only, no decoding/parsing.
for(const key of ['freeze','continuation_ledger']){
  const b=release.upstream[key];const got=bind(b.path);need(got.sha256===b.sha256,'original '+key+' changed');
  fixity.push({...got,role:key,inspection:'hash_only_not_parsed'});
}
const report={schema:'hbe-v5-torsion-failed-attempt-audit-v1',decision:'ACCEPT_FAILED_ATTEMPT_ACCOUNTING',result_status:state.status,
  review_scope:'Saved failure state, ledger, console, receipts and source; two original freeze/ledger files hashed only. No archive, header, measured/predicted curve or numerical result arrays opened.',
  release:terminal.release,source_commit:sourceCommit,declaration:release.declaration,independent_source_review:release.independent_source_review,
  source_files_rehashed_and_git_matched:sourceChecks.length,source_bindings:sourceChecks,outputs:original,finalized_output_bytes:original.reduce((n,f)=>n+f.bytes,0),
  recorded:{readout_calls_attempted:1,held_out_access_attempted:true,held_out_member_reads:null,held_out_responses_accessed:null,ledger_phases:events.map(e=>e.phase),error:state.error,physical_validation_pass:null,empirical_tolerance:null,fit_calls:0,native_calls:0,mesher_calls:0,new_freeze_saved:false,observed_curves_file_present:false,metrics_file_present:false},
  source_trace_inference:{basis:'Authenticated sequential _read_selected reads one entire member before parse_member_csv; traceback reached float conversion. Ledger itself records both intended members before any archive IO, not completed member counts.',
    minimum_complete_member_bodies_materialized:1,maximum_complete_member_bodies_materialized:2,
    necessarily_exposed_first_member:roles.held_out_validation.members[0],
    minimum_uncompressed_member_bytes:roles.held_out_validation.members[0].bytes,maximum_uncompressed_member_bytes:roles.held_out_validation.members.reduce((s,x)=>s+x.bytes,0),
    failed_member_identity:'not recorded; first or second iteration cannot be distinguished',second_member_exposure:'unknown',
    first_member_sha256:'not emitted',numeric_curve_completion_count:'unknown (zero or one)',
    archive_whole_bytes_hash_before_member_loop:'implied completed by source/trace; not independently rehashed by reviewer',
    final_archive_rehash_after_member_loop:'not reached because parse raised',
    warning:'Full CSV body includes response columns even when numeric conversion fails. This is consumed exposure, not zero-access or an unspent validation attempt.'},
  resource_receipts:{worker_elapsed_seconds:stage.elapsed_seconds,parent_elapsed_before_terminal_publication_seconds:terminal.elapsed_before_terminal_publication_seconds,sampled_peak_group_rss_bytes:stage.peak_sampled_process_group_rss_bytes,cleanup:c,resource_limitations:'RSS sampled; parent time excludes final terminal/publication writes.'},
  original_fixity_verified:fixity,other_upstream_identities:release.upstream,
  upstream_limits:'Predictions/numerical bodies not reopened. Their accepted identities and admission follow the saved ledger plus authenticated producer admission. Hash-only current freeze/ledger equality is independently confirmed.',
  next_action:'No retry, header repair or declaration change under this consumed release. Any format-only successor is a separate prospective root decision and must retain this exposure.',
  reviewer_archive_reads:0,reviewer_curve_reads:0,reviewer_native_calls:0,reviewer_model_calls:0,reviewer_reexecution_calls:0};
write(out+'/audit.json',report);
fs.writeFileSync(out+'/REPORT.txt',`Failed attempt accounted; no torsion metric or physical conclusion.\n\nThe sole released readout failed at strict coordinate conversion of the token 'angle'. State retains access_attempted=true and null completed counts. Publication accepted=false correctly binds terminal; worker exit1 was cleanly reaped with no remaining group, fallback or cleanup errors. Worker ${stage.elapsed_seconds} s, parent ${terminal.elapsed_before_terminal_publication_seconds} s before final publication, sampled RSS ${stage.peak_sampled_process_group_rss_bytes} B. Eight finalized files total ${report.finalized_output_bytes} B.\n\nThe ledger records both intended names before archive IO, not two completed reads. Authenticated code plus traceback entails the first negative member's full 1,224 bytes reached the parser; the failure may have occurred in either iteration. One or two full bodies (1,224–2,556 B) were materialized; second-member exposure and exact failed-member identity remain unrecorded. Numeric response conversion is not required for body exposure. Neither zero access nor exactly one completed member is justified.\n\nOriginal freeze and continuation ledger hashes still match. No prediction/numerical/response curve bodies were reopened by this audit. All ${sourceChecks.length} released sources and declaration match historical commit ${sourceCommit}; original strict header=null declaration is unchanged. No metrics, observed-curves export, new freeze, fit, solver or mesher occurred. The attempt remains consumed. Any format-only successor requires a separate prospective root decision.\n`,{flag:'wx'});
const pkg=out+'/compact-failure-evidence-v1';fs.mkdirSync(pkg);const files=[];
function copy(origin,dest){const raw=read(origin);const p=path.join(pkg,dest);fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,raw,{flag:'wx'});files.push({path:dest,origin,bytes:raw.length,sha256:sha(raw)});}
for(const n of names)copy(attempt+'/'+n,'attempt-01/'+n);
for(const n of ['audit.json','REPORT.txt','audit_saved_failure.mjs'])copy(out+'/'+n,n);
copy(terminal.release.path,'release.json');copy(release.independent_source_review.path,'source-review.json');
copy(release.upstream.independent_review.path,'predecessor-admission-review.json');
for(const p of ['scripts/mechanics_hbe_v5_torsion_evaluation.py','scripts/mechanics_hbe_v5_torsion_evaluation_experiment.py','scripts/mechanics_hbe_access.py',release.declaration.path,declaration.roles.path,declaration.protocol.path])copy(p,'source/'+p);
const dependency={source_evidence:{path:'artifacts/hbe-v5-torsion-source-v1/inventory.json',sha256:sha(read('artifacts/hbe-v5-torsion-source-v1/inventory.json'))},upstream:release.upstream,hash_only_fixity:fixity,excluded:'No original archive, selected member contents, prediction/numerical arrays, bulk output trees or original freeze body copied.'};
write(pkg+'/dependencies.json',dependency);files.push({path:'dependencies.json',bytes:fs.statSync(pkg+'/dependencies.json').size,sha256:sha(read(pkg+'/dependencies.json'))});
for(const f of files)need(sha(read(pkg+'/'+f.path))===f.sha256,'package copy');
write(pkg+'/inventory.json',{schema:'hbe-v5-torsion-failure-evidence-v1',files,total_bytes:files.reduce((s,f)=>s+f.bytes,0),dependency:dependency.source_evidence});
for(const f of original)need(sha(read(f.path))===f.sha256,'saved attempt changed during review');
console.log(JSON.stringify({audit:bind(out+'/audit.json'),report:bind(out+'/REPORT.txt'),inventory:bind(pkg+'/inventory.json'),payload_count:files.length,payload_bytes:files.reduce((s,f)=>s+f.bytes,0)},null,2));
