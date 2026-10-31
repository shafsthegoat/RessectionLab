#!/usr/bin/env python3
"""Read-only source-backed release report; writes only catalog/reporting/.

Usable counts come exclusively from catalog/release.json and its checksum-bound
case-decision sidecars. Preparation status is historical, never acceptance.
No imaging arrays or bulk package payloads are opened or rehashed. External
storage is a live NOFOLLOW stat census with AppleDouble metadata separated.
"""
from __future__ import annotations
import argparse,collections,datetime,hashlib,json,os,pathlib,re,stat,sys,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from data_rebuild_inventory import UUID
from resectionlab.data_root import checked_root
DATA=pathlib.Path('/Volumes/shafrir/old coding projects/RessectionLabData');OUT=DATA/'catalog/reporting'
ACCEPT={'accepted_for_stated_capabilities','accepted_with_explicit_limitations'}

def mount():
 # Production fresh native mount/device/filesystem UUID check; no diskutil lease.
 return checked_root(DATA,expected_uuid=UUID)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def family(value,aliases=None):
 s=str(value or '')
 if aliases and s in aliases:s=aliases[s]
 for pattern,prefix,width in [(r'UCSF[-:]PDGM(?:[-:](?:UCSF[-:]PDGM[-:])?)0*(\d+)','UCSF-PDGM:',3),(r'(?:sub-PAT|BTC-PAT)0*(\d+)','BTC:sub-PAT',2),(r'IXI[:_-]?0*(\d+)','IXI:',3),(r'Lausanne[:_-](?:sub-)?0*(\d+)','Lausanne:sub-',3),(r'TopCoW[:_-]0*(\d+)','TopCoW:',3),(r'TopBrain[:_-]0*(\d+)','TopBrain:',3),(r'RESECT[:_-](?:Case)?0*(\d+)','RESECT:Case',0),(r'ReMIND[:_-]0*(\d+)','ReMIND:',3)]:
  m=re.search(pattern,s,re.I)
  if m:return prefix+(str(int(m.group(1))).zfill(width))
 m=re.search(r'ZENODO-17486320.*sub-0*(\d+)',s,re.I)
 return 'ZENODO-17486320:sub-'+str(int(m.group(1))).zfill(2) if m else s

def safe(relative):
 p=pathlib.Path(relative)
 if not p.is_absolute():p=DATA/p
 if '..' in p.parts or not p.absolute().is_relative_to(DATA):raise ValueError('Evidence path outside canonical root')
 current=DATA
 for part in p.relative_to(DATA).parts:
  current/=part
  if current.is_symlink():raise ValueError('Evidence paths must not traverse symlinks')
 return p

class Report:
 def __init__(self):self.evidence=[];self.warnings=[];self.aliases={};self.known_roles={}
 def read(self,relative,default=None):
  p=safe(relative)
  if not p.is_file():return default
  if p.stat().st_size>8*1024*1024:self.warnings.append('Metadata too large for bounded reporting read: '+str(p));return default
  raw=p.read_bytes();self.evidence.append({'path':str(p.relative_to(DATA)),'sha256':sha(raw),'bytes':len(raw),'read_utc':utc()})
  try:return json.loads(raw)
  except (ValueError,UnicodeDecodeError):self.warnings.append('Unreadable JSON evidence: '+str(p));return default
 def text(self,relative):
  p=safe(relative)
  if not p.is_file() or p.stat().st_size>1024*1024:return ''
  raw=p.read_bytes();self.evidence.append({'path':str(p.relative_to(DATA)),'sha256':sha(raw),'bytes':len(raw),'read_utc':utc()});return raw.decode('utf8',errors='replace')
 def published(self,release):
  rows=[];invalid=[];seen=set()
  for case in release.get('cases',[]):
   try:
    if case.get('case_id') in seen:raise ValueError('Duplicate published case identity')
    p=safe(case['path']);review=safe(case['review_path']);raw=review.read_bytes()
    if len(raw)>8*1024*1024 or sha(raw)!=case['review_sha256']:raise ValueError('Linked case-decision checksum differs')
    decision=json.loads(raw);self.evidence.append({'path':str(review.relative_to(DATA)),'sha256':sha(raw),'bytes':len(raw),'read_utc':utc()})
    if case.get('decision') not in ACCEPT or decision.get('independent_decision',{}).get('decision') not in ACCEPT:raise ValueError('Linked decision does not accept stated capabilities')
    if decision.get('case_id')!=case.get('case_id') or str(decision.get('role','')).upper()!=str(case.get('role','')).upper():raise ValueError('Review identity/role differs from release')
    if not case.get('capabilities') or case.get('capabilities')!=decision.get('capabilities') or case.get('capabilities')!=decision.get('independent_decision',{}).get('capabilities'):raise ValueError('Reviewed capabilities differ from release')
    if decision.get('classification')!=case.get('classification'):raise ValueError('Review classification differs from release')
    if decision.get('package_sha256')!=case['sha256'] or decision.get('package_path')!=case['path']:raise ValueError('Decision package binding differs')
    if not p.is_file() or p.stat().st_size!=case['bytes']:raise ValueError('Published package absent or byte count differs')
    role=str(case.get('role','')).upper()
    if any(x in role for x in ['SELECT','EVAL','FROZEN','UNOPENED']):raise ValueError('Protected role found in normal published collection')
    f=family(case.get('person_id',decision.get('family_id')),self.aliases)
    if not f:raise ValueError('Published person-family identity missing')
    if f!=family(decision.get('family_id',decision.get('person_id')),self.aliases):raise ValueError('Review and release family differ after known alias normalization')
    if f in self.known_roles and role!=self.known_roles[f]:raise ValueError('Known person alias role reservation differs from published role')
    classification=str(case.get('classification',''));source=str(case.get('source',''))
    if classification.startswith('synthetic'):kind='synthetic'
    elif f.startswith(('RESECT:','ReMIND:','RHUH:')):kind='surgical_change'
    elif f.startswith(('TopCoW:','TopBrain:')) or 'vascular' in classification.lower():kind='clinical_vascular'
    elif f.startswith('IXI:') or 'control' in classification.lower():kind='healthy_control'
    elif f.startswith('Lausanne:') and f in self.healthy_families:kind='healthy_control'
    elif f.startswith(('UCSF-PDGM:','BTC:','ZENODO-17486320:')):kind='real_tumor'
    else:kind='unclassified_real'
    rows.append({**case,'family_id_normalized':f,'report_category':kind,'license':decision.get('license'),'exposure':decision.get('exposure'),'components':decision.get('components'),'observations':decision.get('observations'),'review_binding_verified':True,'package_hash_recomputed_by_report':False,'production_final_location_reopen':decision.get('production_final_location_reopen'),'native_preparation_status_is_historical':True,'generated_proxy_unit_repairs':decision.get('generated_proxy_unit_repairs',[]),'model_inference_receipts':decision.get('model_inference_receipts',[])})
    seen.add(case['case_id'])
   except Exception as e:invalid.append({'case_id':case.get('case_id'),'reason':str(e)})
  return rows,invalid
 def candidates(self):
  result=[]
  def add(value,origin):
   if isinstance(value,list):
    for x in value:add(x,origin)
   elif isinstance(value,dict):
    if value.get('case_id') and value.get('path') and value.get('sha256'):
     result.append({k:value.get(k) for k in ['case_id','family_id','person_id','path','sha256','classification','role','status','review_status']});result[-1]['evidence']=origin
    for key in ['people','candidates']:
     if key in value:add(value[key],origin)
  files=['catalog/inventory/real-tumor/batch-candidates.json','catalog/inventory/real-tumor/existing-candidates.json','catalog/inventory/surgical-change/RESECT-Case3-candidate.json','catalog/topbrain/batch-candidates.json']
  files+= [str(p.relative_to(DATA)) for p in (DATA/'catalog/topbrain').glob('*-native-candidates.json') if not p.name.startswith('._')]
  files+= [str(p.relative_to(DATA)) for p in (DATA/'catalog/preparation').glob('*.json') if not p.name.startswith('._') and p.name!='run.json']
  for rel in files:add(self.read(rel,{}),rel)
  # Latest explicit recipe revision per stable scenario; historical failed and
  # superseded recipes remain evidence, never extra people or current cases.
  latest={}
  for row in result:
   p=pathlib.Path(row['path']);m=re.search(r'-v(\d+)',p.name);version=int(m.group(1)) if m else 0;key=re.sub(r'-v\d+$','',str(row['case_id']))
   if key not in latest or version>latest[key][0]:latest[key]=(version,row)
  return [r for _,r in latest.values()]

def external_stats():
 started=utc();totals=collections.Counter();groups={};errors=[];stack=[DATA]
 while stack:
  d=stack.pop()
  try:
   with os.scandir(d) as it:
    for item in it:
     p=pathlib.Path(item.path)
     if p==OUT:continue
     try:s=item.stat(follow_symlinks=False)
     except OSError as e:errors.append({'path':str(p),'error':str(e)});continue
     top=p.relative_to(DATA).parts[0];c=groups.setdefault(top,collections.Counter());helper=item.name.startswith('._')
     if stat.S_ISDIR(s.st_mode):c['directories']+=1;stack.append(p)
     elif stat.S_ISLNK(s.st_mode):c['symlinks_not_followed']+=1
     elif stat.S_ISREG(s.st_mode):
      prefix='appledouble' if helper else 'regular';c[prefix+'_files']+=1;c[prefix+'_logical_bytes']+=s.st_size;c[prefix+'_allocated_bytes']+=s.st_blocks*512
      if item.name.endswith(('.partial','.pending','.tmp')):c['in_progress_files']+=1;c['in_progress_logical_bytes']+=s.st_size
     else:c['other_entries']+=1
  except OSError as e:errors.append({'path':str(d),'error':str(e)})
 for c in groups.values():totals.update(c)
 return {'started_utc':started,'finished_utc':utc(),'nofollow':True,'report_outputs_excluded':True,'totals':dict(totals),'by_topdir':groups,'errors':errors,'caveat':'Live stat census while writers may run; directory links never followed. Allocated bytes are st_blocks*512, not promised exclusive/reclaimable storage. AppleDouble metadata helpers are not cases or images.'}

def numeric_resources(value,path=''):
 rows=[]
 if isinstance(value,dict):
  for k,v in value.items():
   current=path+'.'+k if path else k;low=k.lower()
   if isinstance(v,(int,float)) and not isinstance(v,bool) and ('rss' in low or ('memory' in low and any(x in low for x in ['peak','max'])) or k in ['prepare_seconds','production_load_seconds','coldReadyMs','peak_process_rss_bytes','elapsed_seconds']):rows.append({'field':current,'value':v})
   elif isinstance(v,(dict,list)):rows+=numeric_resources(v,current)
 elif isinstance(value,list):
  for i,v in enumerate(value):rows+=numeric_resources(v,path+'['+str(i)+']')
 return rows

def build(stage,scan_stats=True):
 mount();r=Report();release=r.read('catalog/release.json',{});release_sha=r.evidence[-1]['sha256'] if release else None
 ledger=r.read('catalog/work-ledger.json',{});baseline=r.read('catalog/inventory/baseline-patient-groups.json',{});healthy=r.read('catalog/healthy-source-summary.json',{});r.healthy_families={family(x['person']) for x in healthy.get('families',[])}
 alias_records=r.read('catalog/topbrain/person-aliases.json',{});known_recoveries=set()
 for record in alias_records.get('people',[]):
  canonical=family(record['canonical_person_id']);r.aliases.update({alias:canonical for alias in [record['canonical_person_id'],*record.get('aliases',[])]});r.known_roles[canonical]=str(record['role']).upper()
  if record.get('previously_present_in_local_collection'):known_recoveries.add(canonical)
 accepted,invalid=r.published(release);published_hashes={x['sha256'] for x in accepted};families={x['family_id_normalized'] for x in accepted};baseline_families={family(x['family_id']) for x in baseline.get('groups',[])};recovered=(families&({'RESECT:Case3'}|known_recoveries))-baseline_families
 adapters=[]
 for case in accepted:
  try:
   with zipfile.ZipFile(safe(case['path'])) as archive:
    if archive.getinfo('manifest.json').file_size>8*1024*1024:raise ValueError('Oversized preparation metadata')
    contract=json.loads(archive.read('manifest.json')).get('metadata',{}).get('data_case',{})
   if contract.get('ingestion_adapter'):adapters.append({'case_id':case['case_id'],'family_id':case['family_id_normalized'],'policy':contract['ingestion_adapter'],'authority':'Preparation provenance only; acceptance remains linked case-decision'})
  except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:r.warnings.append('Preparation metadata unavailable for '+case['case_id']+': '+str(error))
 categories={}
 for category in ['real_tumor','healthy_control','clinical_vascular','surgical_change','synthetic','unclassified_real']:
  selected=[x for x in accepted if x['report_category']==category];categories[category]={'packages':len(selected),'source_person_families':len({x['family_id_normalized'] for x in selected})}
 hosts={x['family_id_normalized'] for x in accepted if x['family_id_normalized'].startswith(('IXI:','Lausanne:'))};tumors={x['family_id_normalized'] for x in accepted if x['report_category']=='real_tumor' or (x['report_category']=='surgical_change' and x['family_id_normalized'].startswith(('RESECT:','RHUH:')))}
 capabilities=collections.defaultdict(lambda:{'case_ids':[],'families':set()});missing=collections.Counter();components=[]
 for row in accepted:
  for c in row.get('capabilities',[]):capabilities[c]['case_ids'].append(row['case_id']);capabilities[c]['families'].add(row['family_id_normalized'])
  missing.update(set(row.get('missing_anatomy',[])))
  for name,value in (row.get('components') or {}).items():components.append({'case_id':row['case_id'],'component':name,'evidence':value})
 origins=collections.defaultdict(lambda:{'entries':0,'case_ids':set()})
 for component in components:
  value=component['evidence'];provenance=str(value.get('provenance',value.get('kind','unknown')) if isinstance(value,dict) else value).lower()
  origin='generated' if provenance.startswith(('synthetic','generated','simulated')) else 'estimated' if provenance.startswith(('estimated','algorithmic_estimate')) else 'missing' if provenance.startswith(('missing','not_acquired','not_provided','unavailable')) else 'observed_or_supplied' if provenance.startswith(('observed','acquired','source_provided')) else 'unknown'
  component['explicit_origin_category']=origin;origins[origin]['entries']+=1;origins[origin]['case_ids'].add(component['case_id'])
 origin_counts={k:{'component_entries':v['entries'],'accepted_packages':len(v['case_ids']),'case_ids':sorted(v['case_ids'])} for k,v in origins.items()}
 caps={k:{'packages':len(v['case_ids']),'source_person_families':len(v['families']),'case_ids':v['case_ids']} for k,v in capabilities.items()}
 candidates=r.candidates();pending=[x for x in candidates if x['sha256'] not in published_hashes];pending_families={family(x.get('family_id') or x.get('person_id') or x['case_id']) for x in pending}
 decisions=r.read('qc/independent/release-decisions-v1.json',{});review_counts=dict(collections.Counter(x.get('decision','unknown') for x in decisions.get('cases',[])));historical=[]
 for p in (DATA/'qc/independent').glob('*.json'):
  if p.name.startswith('._') or p.name=='release-decisions-v1.json':continue
  d=r.read(str(p.relative_to(DATA)),{})
  if isinstance(d,dict):
   fields={k:v for k,v in d.items() if k in ['decision','status','scale_decision','reason','limitations','reviewer']}
   if any(re.search(r'reject|fail|quarantin|supersed',str(fields.get(k,'')),re.I) for k in ['decision','status','scale_decision']):historical.append({'path':str(p.relative_to(DATA)),**fields})
 proofs=[];resources=[]
 for p in (DATA/'qc/hidden-app').glob('*/report.json'):
  d=r.read(str(p.relative_to(DATA)),{});proofs.append({'path':str(p.relative_to(DATA)),'input':d.get('input'),'input_sha256':d.get('inputSha256'),'passed':d.get('passed'),'window_ever_visible':d.get('windowEverVisible'),'checks':d.get('checks',[]),'started_utc':d.get('startedUtc'),'finished_utc':d.get('finishedUtc'),'bound_to_current_accepted_package':d.get('inputSha256') in published_hashes,'cold_ready_ms':d.get('coldReadyMs')});resources.extend({'path':str(p.relative_to(DATA)),**v} for v in numeric_resources(d))
 for folder in ['catalog/preparation','catalog/topbrain']:
  for p in (DATA/folder).glob('*.json'):
   if p.name.startswith('._'):continue
   d=r.read(str(p.relative_to(DATA)),{});resources.extend({'path':str(p.relative_to(DATA)),**v} for v in numeric_resources(d))
 inference=[]
 for directory,dirs,files in os.walk(DATA/'cache/inference',followlinks=False):
  dirs[:]=[n for n in dirs if not (pathlib.Path(directory)/n).is_symlink() and not n.startswith('._')]
  if 'inference.json' in files:
   p=pathlib.Path(directory)/'inference.json';d=r.read(str(p.relative_to(DATA)),{});inference.append({k:d.get(k) for k in ['case','model_id','status','failure','exit_code','device','elapsed_seconds','peak_process_family_rss_bytes_sampled','memory_caveat','created_at']});inference[-1]['path']=str(p.relative_to(DATA));inference[-1]['saved_output_count']=len(d.get('outputs',[]))
   if d.get('failure'):
    log=r.text(str(p.with_name('inference.log').relative_to(DATA)));facts=[]
    for line in log.splitlines():
     if 'MPS backend out of memory' in line:facts.append(line.split('Use PYTORCH_')[0].strip())
     if 'Signals.SIGKILL: 9' in line:facts.append('Model subprocess terminated by SIGKILL 9; cause beyond recorded termination is not established')
    inference[-1]['recorded_failure_details']=facts;inference[-1]['supervisor_memory_limit_cause_proven']=False
   resources.extend({'path':str(p.relative_to(DATA)),**v} for v in numeric_resources(d))
 guard_probe=r.read('qc/native-volume-guard-probe.json',{});unit_repairs=r.read('qc/proxy-unit-repairs/index.json',{});unit_review=r.read('qc/independent/placement-proxy-unit-repairs-independent-v1.json',{})
 admissions=[]
 for p in (DATA/'qc/mass-effect-resource-admission').glob('*.json'):
  if not p.name.startswith('._'):admissions.append(r.read(str(p.relative_to(DATA)),{}))
 state=r.read('catalog/healthy-batch/state-v5.json',{});plan=r.read('catalog/healthy-batch/plan-v5.json',{});extension=r.read('catalog/healthy-batch/plan-v5-resource-extension.json',{});states=state.get('subjects',{})
 effective_subjects=extension.get('subjects',plan.get('subjects',[]));effective_names={x['subject'] for x in effective_subjects};effective_states={k:v for k,v in states.items() if k in effective_names}
 if extension:
  original=next((x['sha256'] for x in r.evidence if x['path']=='catalog/healthy-batch/plan-v5.json'),None)
  if extension.get('original_plan_sha256')!=original or not extension.get('source_roles_unchanged') or not extension.get('failed_source_retained'):r.warnings.append('Healthy resource extension does not verify original plan/role/retained-failure binding')
 batch={'planned_people':len(effective_subjects),'recorded_people':len(effective_states),'states':dict(collections.Counter(x.get('status','unknown') for x in effective_states.values())),'not_started':max(0,len(effective_subjects)-len(effective_states)),'resource_extension':extension,'retained_original_plan_states_outside_generation_extension':{k:v for k,v in states.items() if k not in effective_names}}
 source_record=r.read('catalog/source-research.json',{});sources=[]
 for s in source_record.get('sources',[]):sources.append({k:s.get(k) for k in ['id','official_url','license','state','access','routes','counts','blocker','blockers','limitations','overlap','split','license_evidence'] if s.get(k) is not None})
 resect=r.read('catalog/inventory/surgical-change/RESECT-Case3-acquisition.json',{})
 if resect and not any('resect' in str(s.get('id','')).lower() for s in sources):
  sources.append({'id':'RESECT','official_url':'https://doi.org/10.11582/2017.00004','license':'CC-BY-4.0 for original images only','state':resect.get('status','pilot_receipt_available'),'counts':{'recovered_already_exposed_TRAIN_families':1,'original_image_files':len(resect.get('files',[]))},'license_evidence':'Frozen original-image manifest plus current official NIRD DOI record; catalog/inventory/surgical-change/README.md','limitations':['Image-only MRI and before/during/after US pilot; registration, masks, cavity/brain-shift/action truth not admitted','Already-exposed TRAIN Case3 recovery adds zero never-seen people']})
 for case in accepted:
  if not any(str(case.get('source','')).lower() in str(x.get('id','')).lower() for x in sources):
   sources.append({'id':case['source'],'license':case.get('license'),'state':'independently accepted package published','limitations':case.get('limitations',[]),'evidence':case['review_path']})
 migration=r.read('catalog/inventory/migration/rehearsal-2026-10-19-migration-receipt.json',{});before=r.read('catalog/inventory/baseline.json',{});after=r.read('catalog/inventory/after-cleanup/baseline.json',{})
 local=lambda d:{k:sum(v.get(k,0) for v in d.get('by_topdir',{}).values()) for k in ['logical_regular_bytes','allocated_lstat_bytes','file','symlink']}
 representatives=[];seen=set()
 for row in accepted:
  key=row['report_category']
  if key in seen:continue
  seen.add(key);p=safe(row['path']);representatives.append({'category':key,'case_id':row['case_id'],'path':str(p),'capabilities':row['capabilities'],'app_action':'Choose configured external Case Library, open this native package; inspect only the independently accepted capabilities.'})
 stats=external_stats() if scan_stats else {'not_run':True}
 current_release=safe('catalog/release.json').read_bytes();changed=sha(current_release)!=release_sha
 result={'schema':'ressectionlab-data-report-v1','generated_utc':utc(),'report_stage':stage,'data_root':str(DATA),'authority':'Published release plus checksum-bound per-case decisions; preparation status is historical. No imaging arrays/bulk packages rehashed by reporting.','release':{'id':release.get('release_id'),'built_utc':release.get('built_utc'),'sha256':release_sha,'changed_while_report_ran':changed,'reported_case_count':release.get('case_count'),'clinical_validation':False},'counts':{'usable_published_packages':len(accepted),'source_person_families_represented':len(families),'observed_real_packages':sum(x['report_category']!='synthetic' for x in accepted),'synthetic_variants':sum(x['report_category']=='synthetic' for x in accepted),'accepted_bounded_mass_effect_fields':sum('bounded_numerical_deformation_inspection' in x.get('capabilities',[]) for x in accepted),'healthy_host_families_represented':len(hosts),'real_tumor_source_families_including_surgical_change':len(tumors),'baseline_existing_families':len(baseline_families),'accepted_baseline_existing_families':len(families&baseline_families),'accepted_already_exposed_case3_recovery':int('RESECT:Case3' in recovered),'accepted_previously_exposed_recovered_families':len(recovered),'accepted_previously_exposed_clinical_recovery':len(recovered&known_recoveries),'accepted_newly_materialized_families_excluding_known_recoveries':len(families-baseline_families-recovered),'accepted_clinical_new_families_excluding_known_recoveries':len({x['family_id_normalized'] for x in accepted if x['report_category']=='clinical_vascular'}-known_recoveries-baseline_families),'accepted_newly_materialized_families_relative_to_baseline_excluding_case3':len(families-baseline_families-recovered),'exact_acquired_tumor_MRI_skull_CT_dedicated_angiography_subjects':release.get('exact_acquired_tumor_MRI_skull_CT_dedicated_angiography_subjects',0),'pending_current_candidate_packages_not_published':len(pending),'pending_source_families':len(pending_families)},'known_person_aliases':alias_records,'accepted_ingestion_adapter_policies':adapters,'recovered_previously_exposed_families':sorted(recovered),'count_definitions':{'real_people':'Declared source person families with known aliases normalized; synthetic variants add zero people. Cross-cohort independence beyond known source aliases is not proven.','newly_materialized':'Absent from initial eight local source groups; not a never-seen/pretraining-independent claim. Case3 is explicitly a previously exposed TRAIN recovery; TopBrain person-aliases independently marks legacy TopCoW001 DEVELOPMENT recovery. Both are excluded from newly materialized family counts.','pending':'Latest preparation descriptors not represented by a valid published package SHA; neither acquisition nor preparation establishes usability. Pending-family counts may overlap accepted families with other accepted variants.','categories':'Package categories are exclusive. Person-family counts across categories may overlap when a host has both real control and synthetic scenarios.'},'categories':categories,'accepted_cases':accepted,'invalid_published_entries':invalid,'pending_candidates':pending,'capabilities':caps,'missing_anatomy_case_counts':dict(missing),'component_provenance':components,'component_origin_counts':origin_counts,'independent_review':{'current_decision_counts':review_counts,'status':decisions.get('status'),'source': 'qc/independent/release-decisions-v1.json','historical_rejections_or_failures':historical,'clinical_or_neuroradiologist_validation':False},'sources':sources,'source_qualification_accessed_utc':source_record.get('accessed_utc'),'application_proofs':proofs,'native_volume_guard_probe':guard_probe,'resource_measurements':resources,'inference_receipts':inference,'mass_effect_resource_admissions':admissions,'mass_effect_plan_and_admission_counts':{'requested_in_plan':sum(bool(x.get('mass_effect')) for x in effective_subjects),'resource_withheld_requests':sum(x.get('requested') is True and x.get('admitted') is False for x in admissions),'accepted_bounded_fields':sum('bounded_numerical_deformation_inspection' in x.get('capabilities',[]) for x in accepted)},'proxy_unit_repair_records':unit_repairs,'proxy_unit_repair_independent_review':unit_review,'source_materialization_receipts':{'healthy_sources':{'state':healthy.get('state'),'counts':healthy.get('counts'),'source_bytes':healthy.get('source_bytes'),'families':healthy.get('families',[])},'real_current_candidate_families':len({family(x.get('family_id') or x.get('person_id') or x['case_id']) for x in candidates if str(x['case_id']).startswith(('UCSF','BTC'))}),'clinical_current_candidate_families':len({family(x.get('family_id') or x.get('person_id') or x['case_id']) for x in candidates if str(x['case_id']).startswith(('TopCoW','TopBrain'))})},'hardware':ledger.get('hardware',{}),'healthy_preparation_batch':batch,'active_jobs_ledger_snapshot':ledger.get('active_jobs',[]),'ledger_status_historical':ledger.get('status'),'known_findings':ledger.get('known_findings',[]),'migration':migration,'local_storage_snapshots':{'before':local(before),'after_scoped_cleanup':local(after),'scope':'Local testcases+data only. Concurrent new small test fixtures mean whole-tree snapshot differences differ from exact retired campaign bytes.'},'external_storage_live_stats':stats,'representatives':representatives,'resume_commands':ledger.get('next_commands',[]),'fixed_commands':{'report_preview':'.venv/bin/python scripts/data_rebuild_report.py --stage preview','report_final_snapshot':'.venv/bin/python scripts/data_rebuild_report.py --stage final','source_case3':'.venv/bin/python scripts/data_rebuild_surgical_change.py','real_train_resume':".venv/bin/python scripts/data_rebuild_real.py batch --count 50 --review-receipt '/Volumes/shafrir/old coding projects/RessectionLabData/qc/independent/UCSF-PDGM-0009-structural-scale-review-v1.json'",'launch_app':'./scripts/launch_poc.sh','environment':"export RESSECTIONLAB_DATA_ROOT='/Volumes/shafrir/old coding projects/RessectionLabData'"},'warnings':r.warnings,'evidence':r.evidence,'completion_claim':'Snapshot only; accepted cases reported separately from ongoing/pending work. This report never marks the research goal complete.'}
 return result

def markdown(d):
 c=d['counts'];lines=[f"# RessectionLab data release — {d['report_stage']} snapshot",'',f"Generated {d['generated_utc']}; published release `{d['release']['id']}` built {d['release']['built_utc']}.",'',f"The published collection currently contains **{c['usable_published_packages']} accepted packages representing {c['source_person_families_represented']} source person families**, including {c['synthetic_variants']} synthetic variants. Synthetic variants add zero people. **{c['pending_current_candidate_packages_not_published']} current candidate packages remain outside the published release**; acquisition and preparation are not acceptance. [1]",'',f"Verified exact same-patient acquired tumor MRI + useful skull CT + dedicated angiography: **{c['exact_acquired_tumor_MRI_skull_CT_dedicated_angiography_subjects']}**. This is engineering/source-data review, with no clinical validation or pretrained-independent evaluation claim.",'', '| Accepted package category | Packages | Source families |','|---|---:|---:|']
 for key,v in d['categories'].items():lines.append(f"| {key} | {v['packages']} | {v['source_person_families']} |")
 lines += ['',f"The initial scoped local inventory identified {c['baseline_existing_families']} source families (one protected frozen assessment family). Current accepted families include {c['accepted_baseline_existing_families']} from that baseline, {c['accepted_newly_materialized_families_relative_to_baseline_excluding_case3']} newly materialized outside it, and {c['accepted_previously_exposed_recovered_families']} previously exposed recoveries (RESECT Case3: {c['accepted_already_exposed_case3_recovery']}; known clinical alias: {c['accepted_previously_exposed_clinical_recovery']}). These are materialization categories, not claims of never-seen patients. Known aliases are normalized; cross-cohort independence remains unresolved. [2]",'', 'Accepted capabilities are limited to the reviewed scopes:','', '| Capability | Packages | Families |','|---|---:|---:|']
 for key,v in sorted(d['capabilities'].items()):lines.append(f"| {key} | {v['packages']} | {v['source_person_families']} |")
 lines+=['','Missing coverage is explicit; acquired images, estimated anatomy and generated scenarios remain distinct in each checksum-bound review. `summary-'+d['report_stage']+'.json` retains complete component evidence and case limitations. [1]','', '| Missing component (case-declared) | Accepted packages |','|---|---:|']
 for k,v in sorted(d['missing_anatomy_case_counts'].items()):lines.append(f'| {k} | {v} |')
 lines+=['','Explicit component provenance (metadata declarations, not anatomical validation; packages can contribute to several rows):','','| Origin | Component entries | Accepted packages |','|---|---:|---:|']
 for key,v in sorted(d['component_origin_counts'].items()):lines.append(f"| {key} | {v['component_entries']} | {v['accepted_packages']} |")
 lines+=['','Source qualification and access [3]:','', '| Source | Recorded state | License | Scope / blocker |','|---|---|---|---|']
 for s in d['sources']:
  note=s.get('blocker') or s.get('blockers') or s.get('limitations') or 'No additional blocker recorded'
  if isinstance(note,list):note='; '.join(str(x) for x in note[:2])
  note=str(note).replace('|','/')
  lines.append(f"| {s['id']} | {s.get('state','unknown')} | {s.get('license','unknown')} | {note} |")
 lines+=['','Independent decisions currently record '+str(d['independent_review']['current_decision_counts'])+'. Historical rejection/failure records and retained limitations are linked in the JSON. AI engineering review is not neuroradiologist validation. Publication still requires an exact accepted case decision; historical native preparation status cannot override it. [4]','',f"Healthy preparation batch: {d['healthy_preparation_batch']['recorded_people']}/{d['healthy_preparation_batch']['planned_people']} planned people have status records; {d['healthy_preparation_batch']['states']}. Ongoing jobs from the ledger are a separate historical status snapshot, not usable-case counts.",'']
 proofs=d['application_proofs'];passed=[x for x in proofs if x['passed'] is True];bound=[x for x in passed if x['bound_to_current_accepted_package']]
 lines += [f"Actual hidden-app reports: {len(passed)}/{len(proofs)} recorded reports passed; {len(bound)} passing reports bind to currently accepted package hashes. Earlier failures remain evidence. These are representative app checks, not proof every candidate received GUI review. All recorded visibility flags and resource values remain in the JSON. [5]",'']
 inferences=d['inference_receipts'];lines += [f"Inference receipts: {len(inferences)} recorded runs, devices {dict(collections.Counter(x.get('device','unknown') for x in inferences))}; failures and retry histories remain in the JSON. No model training was launched by reporting.",'']
 for failure in inferences:
  if '051' in str(failure.get('case','')) and failure.get('failure'):
   details='; '.join(failure.get('recorded_failure_details',[])) or 'No subprocess cause recorded'
   lines += [f"IXI051 {failure.get('device')} attempt: `{failure.get('failure')}`, {failure.get('elapsed_seconds')} seconds, {failure.get('saved_output_count',0)} saved outputs. {details}. This does not establish an8GiB supervisor-limit failure.",'']
 for admission in d['mass_effect_resource_admissions']:
  if admission.get('requested') is True and admission.get('admitted') is False:
   lines += [f"{admission.get('subject')} deformation was withheld before allocation: {admission.get('estimated_additional_array_bytes'):,} estimated additional array bytes exceed {admission.get('budget_bytes'):,} budget bytes. Native geometry/appearance remain available; no silent primary-grid reduction. Accepted bounded fields currently: {d['counts']['accepted_bounded_mass_effect_fields']}.",'']
 repair_rows=d['proxy_unit_repair_independent_review'].get('repairs',[]);verified_repairs=sum(x.get('only_declared_mm_header_changed') is True and x.get('mask_payload_and_sform_exact') is True for x in repair_rows);linked_repairs={x['repaired_sha256'] for case in d['accepted_cases'] for x in case.get('generated_proxy_unit_repairs',[]) if x.get('repaired_sha256')}
 lines += [f"Internal proxy unit repair: {verified_repairs} independently verified header-only mm companions, {len(linked_repairs)} distinct companions linked by current accepted case sidecars. Historical v1 helpers retain unknown units; v2 changes only the declared mm header, preserving mask payload, affine and native anatomy.",'']
 if d['accepted_ingestion_adapter_policies']:
  adapted=sorted({x['family_id'] for x in d['accepted_ingestion_adapter_policies']});lines += [f"Published typed-source adapter policies apply to {', '.join(adapted)}: NIfTI observations and JSON metadata remain distinct; oversized auxiliary views use an explicitly labelled1mm display derivative, while native primary and original files are retained. This is display LOD, not anatomical registration or a replacement acquired scan.",'']

 resources=d['resource_measurements'];rss=[x['value'] for x in resources if 'rss' in x['field'].lower() and 'bytes' in x['field'].lower()]
 if rss:lines += [f"Recorded process RSS samples/peaks range {min(rss)/1024**3:.2f}–{max(rss)/1024**3:.2f} GiB across distinct preparation/app receipts; scopes and units are retained per measurement. This is not a single comparable benchmark or a whole-machine memory peak.",'']
 guard=d['native_volume_guard_probe']
 if guard:lines += [f"Mounted-volume identity uses fresh native Disk Arbitration properties and live filesystem UUID checks on every call, without an identity cache. Recorded probe: {guard.get('samples')} samples, median {1000*guard.get('seconds_median',0):.2f} ms; the current report uses the production guard rather than diskutil.",'']
 m=d['migration']
 if m:lines += [f"Scoped cleanup removed only `{m['original_path']}`: {m['source_logical_bytes_before']:,} logical bytes and {m['source_allocated_lstat_bytes_before']:,} lstat allocated bytes; observed system disk-free change {m['observed_free_change']:+,} bytes. The external tar remains `{m['archive_path']}` (SHA256 `{m['archive_sha256']}`). APFS sharing, snapshots and concurrent work limit interpretation of physical reclamation. Active cases/sources/history were preserved. [6]",'', 'Restore while the original campaign path is absent:','', '```sh',m['restoration_command'],'```','']
 stats=d['external_storage_live_stats']
 if not stats.get('not_run'):
  t=stats['totals'];lines += [f"Live external census: {t.get('regular_logical_bytes',0):,} non-AppleDouble logical bytes plus {t.get('appledouble_logical_bytes',0):,} AppleDouble metadata bytes; {t.get('in_progress_files',0)} in-progress files. NOFOLLOW stat scan {stats['started_utc']}–{stats['finished_utc']}; active writers mean this is not an atomic filesystem snapshot.",'']
 lines+=['Representative accepted packages:','']
 for x in d['representatives']:lines.append(f"- **{x['category']}**: [{x['case_id']}](<{x['path']}>) — {', '.join(x['capabilities'])}.")
 lines+=['','Open with the configured external Case Library after:','','```sh',d['fixed_commands']['environment'],d['fixed_commands']['launch_app'],'```','','Resume current work using the ledger commands in the JSON. Regenerate this report without touching sources, release decisions or the ledger:','','```sh',d['fixed_commands']['report_'+('preview' if d['report_stage']=='preview' else 'final_snapshot')],'```','','Evidence: [1] `catalog/release.json` and each SHA-bound `qc/case-decisions/` sidecar; [2] `catalog/inventory/baseline-patient-groups.json`; [3] `catalog/source-research.json`; [4] `qc/independent/release-decisions-v1.json`; [5] `qc/hidden-app/*/report.json` and preparation receipts; [6] `catalog/inventory/migration/rehearsal-2026-10-19-migration-receipt.json`. Exact metadata checksums are in `evidence-'+d['report_stage']+'.json`.','', 'This is a release snapshot. Pending batches, limitations, unsupported simulation and scientifically unvalidated anatomy are not marked complete.']
 if d['invalid_published_entries']:lines+=['', '**Published-entry binding problems:** '+str(d['invalid_published_entries'])]
 if d['release']['changed_while_report_ran']:lines+=['','**Release changed during generation:** these counts bind to the recorded release SHA; rerun for the latest snapshot.']
 lines += ['', '[1]: <../release.json>', '[2]: <../inventory/baseline-patient-groups.json>', '[3]: <../source-research.json>', '[4]: <../../qc/independent/release-decisions-v1.json>', '[5]: <../../qc/hidden-app/>', '[6]: <../inventory/migration/rehearsal-2026-10-19-migration-receipt.json>']
 return '\n'.join(lines)+'\n'

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['preview','final'],default='preview');ap.add_argument('--skip-storage-stats',action='store_true');a=ap.parse_args();d=build(a.stage,not a.skip_storage_stats);mount();OUT.mkdir(parents=True,exist_ok=True)
 for name,value in [('summary-'+a.stage+'.json',d),('evidence-'+a.stage+'.json',d['evidence'])]:
  p=OUT/name;tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(p)
 p=OUT/('report-'+a.stage+'.md');tmp=p.with_suffix('.md.tmp');tmp.write_text(markdown(d));tmp.replace(p)
 print(json.dumps({'summary':str(OUT/('summary-'+a.stage+'.json')),'report':str(p),'counts':d['counts'],'invalid_published':d['invalid_published_entries'],'release_changed':d['release']['changed_while_report_ran']},indent=2))
if __name__=='__main__':main()
