"""Opt-in independent BTC source-report audit: metadata only, no image imports.

Run with --audit. Fetches the bounded, pinned primary-article archival XML;
opens only named small metadata. Production reader tests use disposable copies.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/btc-six-source-location-independent-v1'
EVIDENCE = 'artifacts/btc-six-source-location-evidence-v1/evidence.json'
CATALOG = 'outputs/patient-corpus-twelve-execution-v1/catalog.json'
DATA = 'data/diffusion_source/ds001226-v5.0.1/'
TSV = DATA + 'participants.tsv'
XML_URI = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6001263/fullTextXML'
XML_SHA = '38e4890b6dc7d7b3bb84e2abfda005f050e7d93cf3c8a9586f74ce0c1a16589d'
DOI = '10.1523/ENEURO.0083-18.2018'
COMMIT = '359d372c5e972a161966312128adb365870df949'
EXPECTED = {
    '05': ('Left', 'Frontal', 16), '16': ('Right', 'Fronto-temporal', 25),
    '20': ('Right', 'Parietal', 28), '22': ('Right', 'Parietal', 29),
    '25': ('Right', 'Temporal', 32), '28': ('Left', 'Frontal', 35),
}
PINS = {
    EVIDENCE: 'f5d9db8e86f7ecc8343b5c302715f93d65727246dfd94a735d5db94d56c64362',
    CATALOG: '5f8b833079e0b4673d2b8d01a36fab474574164755ac731a0972fa8071bcd088',
    TSV: '2ae814b3e90e5c45ea95342a3658e58ba21d3f2d7bf1caa83ef8f1c74986e71d',
    DATA+'README': 'fbb3e2e1b00d699835badc0bbda1296f9a90f2995117d93ba3123bd2f8e2d0bb',
    DATA+'dataset_description.json': '84185f9c387ae232ddbe87e20b329bacf82b9b5dafe9bde63b64c9bc78bed6f2',
}
READER_PINS = {
    'desktop/electron/patient-catalog.cjs': '2383adf1fc42b82d16dec8b193bda2ba5afc3fa45eb2e1f456ecb836aa913556',
    'desktop/electron/patient-catalog.test.cjs': '677633e8fe3ff73e093385a865b205145cefd514d72f365b5e69520e31484bc5',
    'desktop/electron/case-library.cjs': '35d9582eee330bb3b6f860c0bc0c830e6ccca6b9295d8504fc646b87d4d14cf0',
    'desktop/src/WorkbenchPanels.tsx': 'e209704c12b9ba31e792f47883f85eb54a7f201d536bdf5ab7fb5d6bc37147a5',
    'desktop/src/workbench.css': '50d76e51c60aeb8d014a2084a112b39bdea0e4145dbab53d445fad85a6b6cd7f',
    'desktop/tests/case-library-panel.test.mjs': 'db436b43d662b79438a38fe3e16ee2c029e23edb9d29f4b7ee0b23b09ab381db',
}
NODE = '/Users/sayemkamal/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
OPENED = set()


def metadata(relative):
    assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
    path = ROOT / relative
    assert path.name in {'README'} or path.suffix in {'.json', '.tsv', '.py', '.cjs', '.tsx', '.css', '.mjs'}
    assert path.stat().st_size < 2_000_000
    OPENED.add(relative)
    return path.read_bytes()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative):
    assert relative.endswith('.json')
    return json.loads(metadata(relative))


def semantic(value):
    return 'sha256:' + sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def article_rows(raw):
    assert len(raw) < 2_000_000 and sha(raw) == XML_SHA
    xml = ET.fromstring(raw)
    ids = {e.get('pub-id-type'): e.text for e in xml.findall('.//article-meta/article-id')}
    assert ids.get('doi') == DOI and ids.get('pmcid', ids.get('pmc')) in {'6001263', 'PMC6001263'}
    title = ' '.join(''.join(xml.find('.//article-title').itertext()).split())
    assert title == 'Modeling Brain Dynamics in Brain Tumor Patients Using the Virtual Brain'
    table = xml.find(".//table-wrap[@id='T1']")
    assert ''.join(table.find('label').itertext()).strip() == 'Table 1.'
    assert 'Patient characteristics' in ''.join(table.find('caption').itertext())
    headers = [' '.join(''.join(e.itertext()).split()) for e in table.findall('.//thead/tr/th')]
    fields = ('Subject ID', 'Tumor lateralization', 'Tumor location')
    columns = [headers.index(name) for name in fields]
    selected = {}
    for row in table.findall('.//tbody/tr'):
        cells = [' '.join(''.join(e.itertext()).split()) for e in row.findall('td')]
        subject = cells[columns[0]]
        if subject in {'PAT'+n for n in EXPECTED}:
            assert subject not in selected
            selected[subject] = {field: cells[index] for field, index in zip(fields, columns)}
    assert len(selected) == 6
    return selected


def tsv_rows(raw):
    selected = {}
    for line, row in enumerate(csv.DictReader(io.StringIO(raw.decode()), delimiter='\t'), 2):
        subject = row['participant_id']
        if subject in {'sub-PAT'+n for n in EXPECTED}:
            assert subject not in selected
            selected[subject] = (line, {'participant_id': subject, 'tumor location': row['tumor location']})
    assert len(selected) == 6
    return selected


def validate(evidence, catalog, participants, paper, acquisitions):
    assert evidence['schema'] == 'btc-six-source-location-evidence-v1'
    assert evidence['status'] == 'source_reported_unreviewed'
    for key in ('anatomical_review_accepted', 'automatic_seed_placement_allowed', 'canonical_catalog_modified',
                'native_mask_localization_established', 'training_admission_established'):
        assert evidence[key] is False, key
    assert evidence['catalog'] == {'path': CATALOG, 'sha256': PINS[CATALOG], 'catalog_hash': catalog['catalog_hash']}
    source = evidence['source']
    assert source == {'dataset': 'BTC_preop', 'accession': 'ds001226', 'version': '5.0.1', 'commit': COMMIT,
                      'path': TSV, 'sha256': PINS[TSV], 'field': 'tumor location', 'identity_field': 'participant_id',
                      'license': 'CC0', 'uri': f'https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/{COMMIT}/participants.tsv'}
    p = evidence['paper_source']
    assert p['doi'] == DOI and p['pmcid'] == 'PMC6001263' and p['table_id'] == 'T1'
    assert p['sha256'] == XML_SHA and p['metadata_uri'] == XML_URI
    assert p['publication_uri'] == 'https://www.eneuro.org/content/5/3/ENEURO.0083-18.2018/tab-figures-data'
    assert p['selected_fields'] == ['Subject ID', 'Tumor location', 'Tumor lateralization']
    assert p['representation'] == 'Europe PMC archival full-text JATS XML of the primary article; hash is not of publisher HTML'
    assert len(catalog['entries']) == 12
    known = {e['patient_group']: e for e in catalog['entries']}
    assert len(known) == 12
    assert len(evidence['entries']) == 6
    assert {e['participant_id'] for e in evidence['entries']} == {'sub-PAT'+n for n in EXPECTED}
    summaries = []
    for e in evidence['entries']:
        participant = e['participant_id']
        number = participant[-2:]
        side, location, row_number = EXPECTED[number]
        assert e['patient_group'] == 'BTC:'+participant
        c = known[e['patient_group']]
        candidate = c['input_binding']['candidate']
        assert e['role'] == c['role'] == 'TRAIN'
        assert e['case_id'] == 'BTC-ds001226-'+participant+'-preop'
        for key in ('case_id', 'bundle', 'bundle_sha256', 'role', 'source_files', 'acquisition_record'):
            assert e[key] == candidate[key], key
        for key in ('case_id', 'bundle', 'bundle_sha256'):
            assert e[key] == c[key], key
        assert c['input_binding']['bundle_sha256'] == e['bundle_sha256']
        assert candidate['participant'] == {'delimiter':'\t', 'key':'participant_id', 'path':TSV,
                                           'sha256':PINS[TSV], 'value':participant}
        assert all(c[k] is False for k in ('training_admitted', 'training_admission_established', 'clinical_use_permitted', 'virtual_descendant'))
        assert all(c['tumor_location'][k] == 'UNKNOWN' for k in ('anatomical_region', 'hemisphere', 'lobe'))
        assert e['hemisphere'] == e['anatomical_lobe'] == 'UNKNOWN'
        assert e['anatomical_review_accepted'] is e['automatic_seed_placement_allowed'] is False
        assert e['status'] == 'source_reported_unreviewed'
        assert e['source_reported_location'] == location and e['source_reported_lateralization'] == side
        assert e['row_number'] == row_number == participants[participant][0]
        assert e['row_evidence'] == participants[participant][1] == {'participant_id':participant, 'tumor location':location}
        assert e['paper_row_evidence'] == paper['PAT'+number] == {
            'Subject ID':'PAT'+number, 'Tumor location':location, 'Tumor lateralization':side}
        acq = acquisitions[e['acquisition_record']]
        assert all(acq[key] == value for key, value in {'dataset':'BTC_preop', 'accession':'ds001226',
            'release':'5.0.1', 'git_commit':COMMIT, 'subject':participant, 'license':'CC0'}.items())
        acq_files = {f['path']:f for f in acq['files']}
        assert len(acq_files) == len(acq['files'])
        for name in ('participants.tsv', 'README', 'dataset_description.json'):
            assert acq_files[name]['sha256'] == PINS[DATA+name]
        assert len(e['source_files']) == 2 and {f['role'] for f in e['source_files']} == {'imaging','annotation'}
        binding_files = {f['path']:f for f in c['input_binding']['metadata_and_sources']}
        for f in e['source_files']:
            assert f['path'].startswith(DATA)
            assert acq_files[f['path'][len(DATA):]]['sha256'] == f['sha256'] == binding_files[f['path']]['sha256']
        summaries.append({'participant_id':participant,'patient_group':e['patient_group'],'case_id':e['case_id'],
            'case_hash':c['case_hash'],'catalog_bound_bundle_sha256':e['bundle_sha256'],
            'source_files':e['source_files'],'acquisition_record':e['acquisition_record'],
            'acquisition_sha256':binding_files[e['acquisition_record']]['sha256'],
            'tsv_row_number':row_number,'source_reported_lateralization':side,'source_reported_location':location,
            'canonical_hemisphere':'UNKNOWN','canonical_lobe':'UNKNOWN','training_admitted':False})
    return summaries


def tamper_controls(evidence, catalog, participants, paper, acquisitions):
    base = [evidence, catalog, participants, paper, acquisitions]
    checks = []
    def reject(name, change):
        values = copy.deepcopy(base)
        change(values)
        try:
            validate(*values)
        except (AssertionError, KeyError):
            checks.append(name)
        else:
            raise AssertionError('accepted mutation: '+name)
    for key in ('anatomical_review_accepted', 'automatic_seed_placement_allowed', 'canonical_catalog_modified',
                'native_mask_localization_established', 'training_admission_established'):
        reject('top_'+key, lambda v,k=key: v[0].__setitem__(k,True))
    for key,value in {'participant_id':'sub-PAT26','patient_group':'BTC:sub-PAT26','role':'SELECT','case_id':'other',
                      'bundle':'outputs/other.ressectionlab','bundle_sha256':'0'*64,'acquisition_record':'other.json',
                      'row_number':17,'source_reported_location':'Occipital','source_reported_lateralization':'Right',
                      'hemisphere':'Left','anatomical_lobe':'Frontal','anatomical_review_accepted':True,
                      'automatic_seed_placement_allowed':True,'status':'accepted'}.items():
        reject('row_'+key, lambda v,k=key,x=value: v[0]['entries'][0].__setitem__(k,x))
    reject('duplicate_identity', lambda v: v[0]['entries'].__setitem__(1,copy.deepcopy(v[0]['entries'][0])))
    reject('extra_identity', lambda v: v[0]['entries'].append(copy.deepcopy(v[0]['entries'][0])))
    reject('wrong_source_sha', lambda v: v[0]['entries'][0]['source_files'][0].__setitem__('sha256','0'*64))
    reject('paper_subject_mapping', lambda v: v[3]['PAT05'].__setitem__('Subject ID','PAT28'))
    reject('paper_laterality', lambda v: v[3]['PAT05'].__setitem__('Tumor lateralization','Right'))
    reject('tsv_location', lambda v: v[2]['sub-PAT05'][1].__setitem__('tumor location','Occipital'))
    reject('acquisition_subject', lambda v: v[4][v[0]['entries'][0]['acquisition_record']].__setitem__('subject','sub-PAT26'))
    reject('acquisition_source_sha', lambda v: next(f for f in v[4][v[0]['entries'][0]['acquisition_record']]['files'] if f['path'].endswith('.nii.gz')).__setitem__('sha256','0'*64))
    reject('catalog_location_promotion', lambda v: v[1]['entries'][0]['tumor_location'].__setitem__('lobe','Frontal'))
    reject('catalog_training_admission', lambda v: v[1]['entries'][0].__setitem__('training_admitted',True))
    return checks


NODE_CHECK = r"""
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=process.cwd(); let reads=0;
for(const name of ['open','readFile']){const old=fs[name];fs[name]=async function(file,...args){
  if(typeof file==='string'){assert(!/\.(nii(?:\.gz)?|ressectionlab|npz|npy)$/i.test(file),'payload read attempted');reads++;}
  return old.call(this,file,...args);
};}
const {PatientCatalog,RELEASE,TWELVE_RELEASE,LOCATION_EVIDENCE,validateSourceLocationEvidence}=require(path.join(root,'desktop/electron/patient-catalog.cjs'));
(async()=>{
 const raw=await fs.readFile(path.join(root,LOCATION_EVIDENCE.path)),evidence=JSON.parse(raw),catalog=JSON.parse(await fs.readFile(path.join(root,TWELVE_RELEASE.catalogPath)));
 const original=JSON.stringify(catalog),expected=evidence.entries.map(e=>[e.patient_group,{label:e.source_reported_lateralization+' '+e.source_reported_location,status:'source-reported-unreviewed'}]);
 const real=await new PatientCatalog(root).load();assert.equal(real.length,12);assert.deepEqual(real.filter(r=>r.sourceLocation).map(r=>[r.entry.patient_group,r.sourceLocation]),expected);
 assert(real.every(r=>['lobe','hemisphere','anatomical_region'].every(k=>r.entry.tumor_location[k]==='UNKNOWN')));
 let rejected=0;
 const mutations=[
  e=>e.anatomical_review_accepted=true,e=>e.automatic_seed_placement_allowed=true,e=>e.native_mask_localization_established=true,
  e=>e.training_admission_established=true,e=>e.entries[0].role='SELECT',e=>e.entries[0].case_id='other',
  e=>e.entries[0].bundle_sha256='0'.repeat(64),e=>e.entries[0].source_files[0].sha256='0'.repeat(64),
  e=>e.entries[0].acquisition_record='other.json',e=>e.entries[0].source_reported_lateralization='Right',
  e=>e.entries[0].anatomical_lobe='Frontal',e=>e.entries[0].hemisphere='Left',
  e=>e.entries[0].paper_row_evidence['Subject ID']='PAT28',e=>e.entries.push(e.entries[0]),e=>e.entries[1]=e.entries[0]
 ];
 for(const change of mutations){const e=structuredClone(evidence);change(e);assert.throws(()=>validateSourceLocationEvidence(e,catalog.entries));rejected++;}
 assert.equal(JSON.stringify(catalog),original);
 const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'btc-independent-'));
 try {
  const fixture=await fs.realpath(temporary),names=[RELEASE.path,RELEASE.catalogPath,TWELVE_RELEASE.path,TWELVE_RELEASE.prospectivePath,TWELVE_RELEASE.catalogPath,LOCATION_EVIDENCE.path];
  for(const name of names){await fs.mkdir(path.dirname(path.join(fixture,name)),{recursive:true});await fs.copyFile(path.join(root,name),path.join(fixture,name));}
  const reader=new PatientCatalog(fixture),before=await reader.load();assert.equal(before.length,12);
  await fs.writeFile(path.join(fixture,LOCATION_EVIDENCE.path),Buffer.concat([raw,Buffer.from(' ')]));await assert.rejects(reader.load(),/checksum/);
  await fs.writeFile(path.join(fixture,LOCATION_EVIDENCE.path),raw);await fs.unlink(path.join(fixture,LOCATION_EVIDENCE.path));
  const absent=await reader.load();assert.deepEqual(absent.map(r=>r.id),before.map(r=>r.id));assert(absent.every(r=>!r.sourceLocation));
  await fs.writeFile(path.join(fixture,LOCATION_EVIDENCE.path),raw);await fs.unlink(path.join(fixture,TWELVE_RELEASE.catalogPath));
  const ten=await reader.load();assert.equal(ten.length,10);assert(ten.every(r=>!r.sourceLocation));
  await fs.writeFile(path.join(fixture,LOCATION_EVIDENCE.path),'{}');await assert.rejects(reader.load(),/checksum/);
  await fs.writeFile(path.join(fixture,LOCATION_EVIDENCE.path),raw);await fs.unlink(path.join(fixture,RELEASE.catalogPath));
  assert.deepEqual(await reader.load(),[]);
  await fs.writeFile(path.join(fixture,LOCATION_EVIDENCE.path),'{}');await assert.rejects(reader.load(),/checksum/);
  await fs.unlink(path.join(fixture,LOCATION_EVIDENCE.path));await fs.symlink(path.join(root,LOCATION_EVIDENCE.path),path.join(fixture,LOCATION_EVIDENCE.path));
  await assert.rejects(reader.load(),/escaped/);
 }finally{await fs.rm(temporary,{recursive:true,force:true});}
 console.log(JSON.stringify({status:'pass',actual_rows:real.length,reported_sites:expected,pure_mutations_rejected:rejected,
   integration_checks:['exact_real_metadata_labels','canonical_unknown_preserved','whitespace_tamper_checksum_rejection','absent_evidence_stable_ids_no_labels','ten_fallback_no_labels','bad_present_evidence_rejected_on_ten','no_catalog_empty','bad_present_evidence_rejected_without_catalog','redirected_evidence_rejected'],
   metadata_reads:reads,payload_reads:0,resolve_or_bundle_open_invoked:false}));
})().catch(error=>{console.error(error);process.exitCode=1});
"""


def audit():
    frozen = {**PINS, **READER_PINS}
    for path, pin in frozen.items():
        assert sha(metadata(path)) == pin, path
    catalog, evidence = read(CATALOG), read(EVIDENCE)
    assert catalog['catalog_hash'] == semantic({k:v for k,v in catalog.items() if k != 'catalog_hash'})
    receipts = []
    for entry in catalog['entries']:
        name = str(Path(CATALOG).parent / entry['receipt_path'])
        receipt = read(name)
        assert receipt == {k:v for k,v in entry.items() if k != 'receipt_path'}
        assert receipt['receipt_hash'] == semantic({k:v for k,v in receipt.items() if k != 'receipt_hash'})
        assert receipt['input_binding_hash'] == semantic(receipt['input_binding'])
        frozen[name] = sha(metadata(name))
        receipts.append({'patient_group':entry['patient_group'], 'receipt_sha256':frozen[name]})
    acquisitions = {}
    for e in evidence['entries']:
        row = next(r for r in catalog['entries'] if r['patient_group'] == e['patient_group'])
        pin = next(f['sha256'] for f in row['input_binding']['metadata_and_sources'] if f['path'] == e['acquisition_record'])
        assert sha(metadata(e['acquisition_record'])) == pin
        frozen[e['acquisition_record']] = pin
        acquisitions[e['acquisition_record']] = read(e['acquisition_record'])
    readme = metadata(DATA+'README').decode()
    description = read(DATA+'dataset_description.json')
    assert 'PMC6001263' in readme and DOI in readme
    assert DOI in json.dumps(description['ReferencesAndLinks']) and description['License'] == 'CC0'
    request = urllib.request.Request(XML_URI, headers={'User-Agent':'RessectionLab-independent-metadata-audit/1'})
    with urllib.request.urlopen(request, timeout=45) as response:
        assert response.status == 200
        xml_bytes = response.read(2_000_001)
    paper = article_rows(xml_bytes)
    participants = tsv_rows(metadata(TSV))
    summaries = validate(evidence,catalog,participants,paper,acquisitions)
    controls = tamper_controls(evidence,catalog,participants,paper,acquisitions)
    reader = subprocess.run([NODE,'-'], input=NODE_CHECK, text=True,cwd=ROOT,capture_output=True,timeout=45)
    assert reader.returncode == 0, reader.stderr
    reader_result = json.loads(reader.stdout)
    for path, pin in frozen.items():
        assert sha(metadata(path)) == pin, ('changed during audit',path)
    return {'schema':'btc-six-source-location-independent-qa-v1','status':'pass_metadata_and_reader_only',
        'auditor_sha256':sha(metadata(str(Path(__file__).relative_to(ROOT)))),
        'pinned_metadata_unchanged_before_after':frozen,'verified_catalog_receipts':receipts,
        'primary_source':{'metadata_uri':XML_URI,'xml_sha256':sha(xml_bytes),'xml_bytes':len(xml_bytes),
            'doi':DOI,'pmcid':'PMC6001263','table_id':'T1','header_name_based_extraction':True,
            'dataset_readme_names_exact_article':True,'selected_rows':list(paper.values())},
        'verified_six_source_bindings':summaries,'independent_mutations_rejected':controls,'reader':reader_result,
        'canonical_catalog_changed':False,'native_mask_location_accepted':False,'anatomical_review_accepted':False,
        'automatic_seed_placement_allowed':False,'training_admission_established':False,
        'source_arrays_or_bundles_opened':False,'models_or_solvers_executed':False,'gui_observed':False,
        'metadata_files_read':sorted(OPENED),
        'limitations':['Patient-level reported site is not localization of any native annotation, geometry or seed.',
            'Bundle and image hashes are cross-checked across frozen metadata; no image or bundle bytes were opened or rehashed.',
            'Paper identity mapping uses exact dataset citation, PATnn identifiers and selected TSV site agreement; no anatomy review.',
            'TRAIN is the existing development partition; training admission remains false.',
            'Reader checks are non-GUI; actual native UI observation belongs to a separate root-owned check.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true', required=True)
    parser.parse_args()
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'independent-qa.json'
    with target.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'report':str(target.relative_to(ROOT)),
                      'sha256':sha(target.read_bytes()),'tamper_controls':len(result['independent_mutations_rejected'])},indent=2))
