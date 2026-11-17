#!/usr/bin/env python3
"""Audit and pair the diagnostic with byte-bound original measurements."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,statistics,subprocess,time,html
from pathlib import Path
from run_fixed_preparation_diagnostic import ROOT,ORIGINAL,ORIGINAL_COMMIT,DEFAULT,freeze_bundle,load_original,check_original_immutable,read,save,sha
FAMILIES={0:'narrow-preparation-gated',1:'gated-critical-residual',2:'permitted-overlap-budget',3:'critical-residual'}
def mean(xs):
 values=[x for x in xs if x is not None];return statistics.mean(values) if values else None
def csvsave(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,default=DEFAULT);args=p.parse_args();out=args.evidence.resolve()
 bundle,manifest=freeze_bundle(out);load_original(bundle);check_original_immutable(manifest)
 diagnostic=read(out/'evaluation.json');original=read(ORIGINAL/'evaluation.json');protocol=read(ORIGINAL/'protocol.json')
 if diagnostic['sourceBundle']['sha256']!=sha(bundle/'bundle.json'):raise ValueError('Bundle binding differs')
 if diagnostic['originalProtocol']['sha256']!=sha(ORIGINAL/'protocol.json') or diagnostic['originalEvaluation']['sha256']!=sha(ORIGINAL/'evaluation.json'):raise ValueError('Original bindings differ')
 expected={(tier,parent) for tier in ('low','high') for parent in range(200,216)}
 actual=[(r['tier'],r['parent']) for r in diagnostic['rows']]
 if len(actual)!=32 or set(actual)!=expected or len(set(actual))!=32:raise ValueError('Exact32unique diagnostic jobs required')
 selected={(r['tier'],r['parent']):r for r in diagnostic['rows']}
 evidence_roots={key:out for key in selected};attempts=list(diagnostic['rows']);retry_binding=None
 retry_root=out/'clean-attempt-v2'
 if not (retry_root/'evaluation.json').exists():raise ValueError('Declared clean timing retries must finish before primary analysis')
 retry=read(retry_root/'evaluation.json');retry_plan=read(retry_root/'plan.json')
 if retry['initialEvaluationSha256']!=sha(out/'evaluation.json') or retry['planSha256']!=sha(retry_root/'plan.json'):raise ValueError('Retry binding differs')
 if retry['controllerSha256']!=diagnostic['controllerSha256'] or retry['sourceBundleSha256']!=diagnostic['sourceBundle']['sha256']:raise ValueError('Retry changed controller/environment')
 retry_keys=[(r['tier'],r['parent']) for r in retry['rows']]
 expected_retry={(r['tier'],r['parent']) for r in retry_plan['jobs']}
 if len(retry_keys)!=len(expected_retry) or set(retry_keys)!=expected_retry:raise ValueError('Retry jobs differ from predeclared timing windows')
 for r in retry['rows']:
  key=(r['tier'],r['parent']);selected[key]=r;evidence_roots[key]=retry_root
 attempts+=retry['rows'];retry_binding=dict(path=str(retry_root/'evaluation.json'),sha256=sha(retry_root/'evaluation.json'),planSha256=sha(retry_root/'plan.json'),initialAttemptsRetained=True)
 primary_rows=[selected[(r['tier'],r['parent'])] for r in diagnostic['rows']]
 original_backend=read(ORIGINAL/'execution-environment.json')['backend']
 if original_backend['binarySha256']!=diagnostic['nativeBackend']['binarySha256']:raise ValueError('Original compiled binary changed')
 for key in ('python','torch','threads','device'):
  if key in protocol['hardware'] and diagnostic['hardware'][key]!=protocol['hardware'][key]:raise ValueError('Original timing setup differs: '+key)
 # Reuse only immutable byte-bound prior physical measurements; timing remains
 # historical, not a newly matched speed trial. No old result is rewritten.
 old_csv=ROOT/'docs/sequential-milestone-results.csv'
 expected_old_csv=hashlib.sha256(subprocess.check_output(['git','show',ORIGINAL_COMMIT+':docs/sequential-milestone-results.csv'],cwd=ROOT)).hexdigest()
 if sha(old_csv)!=expected_old_csv:raise ValueError('Original result CSV changed')
 prior=list(csv.DictReader(old_csv.open()));prior_rows=[]
 for c in prior:
  r=next(r for r in original['rows'] if (r['tier'],r['parent'],r['method'],r['seed'])==(c['tier'],int(c['parent']),c['method'],int(c['seed'])))
  if sha(c['artifact'])!=c['artifact_sha256'] or read(c['artifact'])!=r:raise ValueError('Original row bytes changed')
  for key in c:
   if key in ('parent','pattern','seed'):c[key]=int(c[key])
   elif c[key] in ('True','False'):c[key]=c[key]=='True'
   elif c[key]=='':c[key]=None
   else:
    try:c[key]=float(c[key])
    except (ValueError,TypeError):pass
  c['modeled_seconds']=r['snapshot']['checkpoint']['state']['elapsedSeconds'];prior_rows.append(c)
 # The original summary's independent force-replay/compaction is reused as
 # exact source bytes, imported against the archived original package.
 original_summary=out/'analysis-tools/original_summarizer.py';original_summary.parent.mkdir(exist_ok=True)
 summary_bytes=subprocess.check_output(['git','show',ORIGINAL_COMMIT+':scripts/summarize_sequential_milestone.py'],cwd=ROOT)
 if original_summary.exists() and original_summary.read_bytes()!=summary_bytes:raise ValueError('Original analysis tool changed')
 original_summary.write_bytes(summary_bytes)
 spec=importlib.util.spec_from_file_location('diagnostic_original_summary',original_summary);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 audit=s.NativeAudit();started=time.monotonic();new=[]
 for r in primary_rows:
  c=s.compact(r,evidence_roots[r['tier'],r['parent']],audit);c['modeled_seconds']=r.get('snapshot',{}).get('checkpoint',{}).get('state',{}).get('elapsedSeconds')
  new.append(c)
 allrows=new+prior_rows
 for r in allrows:
  r['family']=FAMILIES[r['pattern']];r['controller']=r['method']+(f"-s{r['seed']}" if r['method']=='learned' else '')
 metrics=['treatment_fraction','qualified_treatment_fraction','treated_mm3','residual_mm3','declared_task_complete','valid_stop_withdrawal','neural_removed_mm3','motor_cortex_removed_mm3','cst_removed_mm3','brainstem_removed_mm3','cerebellum_removed_mm3','neural_cut_source_mm3','motor_cortex_cut_source_mm3','cst_cut_source_mm3','bone_mm3','debris_mm3','fluid_mm3','actual_attempts','actual_refusals','preview_refusals','physical_requests','planning_requests','execution_requests','canceled_requests','online_wall_seconds','modeled_seconds']
 grouped=[]
 for tier in ('low','high'):
  for family in FAMILIES.values():
   for controller in sorted({r['controller'] for r in allrows}):
    g=[r for r in allrows if r['tier']==tier and r['family']==family and r['controller']==controller]
    if len(g)!=4:raise ValueError('Four layouts per controller/family required')
    grouped.append(dict(tier=tier,family=family,controller=controller,runs=len(g),**{k:mean([r[k] for r in g]) for k in metrics},failed=sum(bool(r['failure']) for r in g),unqualified=sum(not r['valid_stop_withdrawal'] for r in g)))
 paired=[]
 for a in new:
  for b in prior_rows:
   if (a['parent'],a['tier'])!=(b['parent'],b['tier']):continue
   paired.append(dict(parent=a['parent'],family=a['family'],tier=a['tier'],baseline=b['controller'],**{
    k+'_difference':float(a[k])-float(b[k]) if a[k] is not None and b[k] is not None else None for k in metrics},
    timing='New measured control minus historical original; not a matched retiming trial'))
 assert len(paired)==160
 paired_groups=[]
 for tier in ('low','high'):
  for family in FAMILIES.values():
   for baseline in sorted({r['baseline'] for r in paired}):
    g=[r for r in paired if (r['tier'],r['family'],r['baseline'])==(tier,family,baseline)]
    paired_groups.append(dict(tier=tier,family=family,baseline=baseline,pairs=len(g),**{k+'_difference':mean([r[k+'_difference'] for r in g]) for k in metrics}))
 csvsave(out/'rows.csv',allrows);csvsave(out/'family-summary.csv',grouped);csvsave(out/'paired.csv',paired);csvsave(out/'paired-family-summary.csv',paired_groups)
 totals={k:dict(known=sum(r[k] for r in new if r[k] is not None),missing=sum(r[k] is None for r in new)) for k in ('physical_requests','planning_requests','execution_requests','canceled_requests','actual_attempts','actual_refusals','preview_refusals','online_wall_seconds')}
 claims=[]
 for tier in ('low','high'):
  a=[r for r in new if r['tier']==tier];pairs=[r for r in paired if r['tier']==tier and r['baseline']=='learned-s17']
  claims.append(dict(tier=tier,controlMeanTreatment=mean([r['qualified_treatment_fraction'] for r in a]),
   learnedSeed17MeanTreatment=mean([r['qualified_treatment_fraction'] for r in prior_rows if r['tier']==tier and r['controller']=='learned-s17']),
   controlMinusSeed17=mean([r['qualified_treatment_fraction_difference'] for r in pairs]),
   controlSeed17Wins=sum(r['qualified_treatment_fraction_difference']>1e-10 for r in pairs),
   controlSeed17Ties=sum(abs(r['qualified_treatment_fraction_difference'])<=1e-10 for r in pairs),
   controlSeed17Losses=sum(r['qualified_treatment_fraction_difference']< -1e-10 for r in pairs),
   delayedControlMinusSeed17=mean([r['qualified_treatment_fraction_difference'] for r in pairs if r['family']==FAMILIES[0]])))
 check_original_immutable(manifest)
 report=dict(schema='resectionlab.fixed-preparation-diagnostic-results.v1',scope=diagnostic['scope'],
  interpretation='The costed fixed-preparation control challenges a fixed-choice explanation; report each tier/family. Existing inspected layouts provide diagnostic evidence only.',
  sourceCommit=ORIGINAL_COMMIT,sourceBundle=dict(path=str(bundle),sha256=sha(bundle/'bundle.json')),
  evaluation=dict(path=str(out/'evaluation.json'),sha256=sha(out/'evaluation.json')),
  sourceHashes={p:sha(ROOT/p) for p in ['scripts/run_fixed_preparation_diagnostic.py','src/resectionlab/surgical_fixed_preparation_control.py','scripts/summarize_fixed_preparation_diagnostic.py']},
  originalMeasurements=dict(protocolSha256=sha(ORIGINAL/'protocol.json'),evaluationSha256=sha(ORIGINAL/'evaluation.json'),resultsCsvSha256=sha(old_csv),unchanged=True),
  rows=32,pairedComparisons=160,families=FAMILIES,claims=claims,onlineCostTotals=totals,
  timingRetry=retry_binding,recordedAttempts=len(attempts),
  allAttemptCostTotals={k:sum(r.get('computation',{}).get(k,0) for r in attempts) for k in ('physicalCalls','planningPhysicalCalls','executionPhysicalCalls','canceledRequests','wallSeconds')},
  replayRequests=sum(r.get('replay',{}).get('physicalCalls',0) for r in attempts),
  replaySeconds=sum(r.get('replay',{}).get('wallSeconds',0) for r in attempts),
  independentAcceptedHistoryAudit=dict(uniqueSnapshots=len(audit.cache),physicalRequests=audit.requests,wallSeconds=audit.seconds,failures=audit.failures),
  analysisWallSeconds=time.monotonic()-started,failures=diagnostic['failures']+retry['failures'],unqualified=sum(not r['valid_stop_withdrawal'] for r in new),
  unknownEvidence=sorted({v for r in diagnostic['rows'] for v in r.get('missingEvidence',[])}),
  limits=['No new untouched evaluation/generalization or clinical claim','No fitting/optimizer updates','Original compiled binary/code used; comparator wall times are historical','Unknown earlier access costs, functional consequences and unsupported physics remain unknown'],
  artifacts={n:dict(path=str(out/n),sha256=sha(out/n)) for n in ['rows.csv','family-summary.csv','paired.csv','paired-family-summary.csv']})
 save(out/'summary.json',report)
 # Local inspectable index; native replay workspace is delivered separately.
 headers=['tier','family','controller','qualified_treatment_fraction','declared_task_complete','valid_stop_withdrawal','neural_removed_mm3','motor_cortex_removed_mm3','cst_removed_mm3','actual_refusals','preview_refusals','physical_requests','online_wall_seconds']
 def table(rs,ks):return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in ks)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ks)+'</tr>' for r in rs)+'</tbody></table>'
 (out/'index.html').write_text('<!doctype html><meta charset=utf-8><title>Fixed preparation diagnostic</title><style>body{font:14px system-ui;max-width:1500px;margin:30px auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>Costed fixed0.4mm preparation, then unchanged greedy</h1><p>Diagnostic reuse of previously inspected layouts. Same original compiled engine, source, harm/endpoint rules and low/high budgets. Preparation, previews, rejected requests and STOP are charged. Prior wall times are historical; no matched retiming claim. No training or patient execution.</p><h2>Paired challenge to seed17</h2><pre>'+html.escape(json.dumps(claims,indent=2))+'</pre><h2>Every geometry family and controller</h2>'+table(grouped,headers)+'<h2>Exact provenance, costs and unknowns</h2><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>')
 save(out/'summary.json',dict(report,index=dict(path=str(out/'index.html'),sha256=sha(out/'index.html'))))
 print(json.dumps(dict(claims=claims,unqualified=report['unqualified'],failures=report['failures'],costs=totals),indent=2))
if __name__=='__main__':main()
