"""Verify six source-reported sites using metadata only; never open imaging."""
from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CATALOG='outputs/patient-corpus-twelve-execution-v1/catalog.json'
CATALOG_SHA='5f8b833079e0b4673d2b8d01a36fab474574164755ac731a0972fa8071bcd088'
SOURCE='data/diffusion_source/ds001226-v5.0.1/participants.tsv'
SOURCE_SHA='2ae814b3e90e5c45ea95342a3658e58ba21d3f2d7bf1caa83ef8f1c74986e71d'
COMMIT='359d372c5e972a161966312128adb365870df949'
PAPER_URI='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6001263/fullTextXML'
PAPER_SHA='38e4890b6dc7d7b3bb84e2abfda005f050e7d93cf3c8a9586f74ce0c1a16589d'
# Explicit participant rows in primary paper Table 1; never an image inference.
FACTS={'05':('Frontal','Left'),'16':('Fronto-temporal','Right'),'20':('Parietal','Right'),
    '22':('Parietal','Right'),'25':('Temporal','Right'),'28':('Frontal','Left')}


def need(condition,message):
    if not condition:raise ValueError(message)


def pinned(path,expected):
    body=path.read_bytes();need(sha256(body).hexdigest()==expected,'Pinned metadata changed: '+str(path));return body


def expected(root=ROOT):
    catalog=json.loads(pinned(root/CATALOG,CATALOG_SHA))
    rows={}
    for number,row in enumerate(csv.DictReader(io.StringIO(pinned(root/SOURCE,SOURCE_SHA).decode()),delimiter='\t'),2):
        subject=row['participant_id']
        if subject in {'sub-PAT'+n for n in FACTS}:
            need(subject not in rows,'Duplicate eligible participant row')
            rows[subject]=(number,{'participant_id':subject,'tumor location':row['tumor location']})
    entries=[]
    for number,(location,laterality) in FACTS.items():
        subject='sub-PAT'+number;group='BTC:'+subject
        selected=[row for row in catalog['entries'] if row['patient_group']==group]
        need(len(selected)==1 and selected[0]['role']=='TRAIN','Eligible case identity/role changed')
        case=selected[0];candidate=case['input_binding']['candidate'];line,row=rows[subject]
        need(row['tumor location']==location,'TSV and primary paper location disagree')
        need(candidate['participant']['sha256']==SOURCE_SHA and candidate['participant']['value']==subject,
            'Candidate participant source binding changed')
        entries.append({'patient_group':group,'case_id':case['case_id'],'role':case['role'],'bundle':case['bundle'],
            'bundle_sha256':case['bundle_sha256'],'source_files':candidate['source_files'],
            'acquisition_record':candidate['acquisition_record'],'participant_id':subject,'row_number':line,'row_evidence':row,
            'paper_row_evidence':{'Subject ID':'PAT'+number,'Tumor location':location,'Tumor lateralization':laterality},
            'source_reported_location':location,'source_reported_lateralization':laterality,
            'status':'source_reported_unreviewed','hemisphere':'UNKNOWN','anatomical_lobe':'UNKNOWN',
            'anatomical_review_accepted':False,'automatic_seed_placement_allowed':False})
    return {'schema':'btc-six-source-location-evidence-v1','status':'source_reported_unreviewed',
        'catalog':{'path':CATALOG,'sha256':CATALOG_SHA,'catalog_hash':catalog['catalog_hash']},
        'source':{'dataset':'BTC_preop','accession':'ds001226','version':'5.0.1','commit':COMMIT,'path':SOURCE,
            'uri':f'https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/{COMMIT}/participants.tsv',
            'sha256':SOURCE_SHA,'field':'tumor location','identity_field':'participant_id','license':'CC0'},
        'paper_source':{'publication_uri':'https://www.eneuro.org/content/5/3/ENEURO.0083-18.2018/tab-figures-data',
            'doi':'10.1523/ENEURO.0083-18.2018','pmcid':'PMC6001263','metadata_uri':PAPER_URI,'sha256':PAPER_SHA,
            'representation':'Europe PMC archival full-text JATS XML of the primary article; hash is not of publisher HTML',
            'table_id':'T1','table_label':'Table 1. Patient characteristics',
            'selected_fields':['Subject ID','Tumor location','Tumor lateralization'],
            'identity_mapping':'Paper PATnn maps to BTC_preop sub-PATnn; the pinned dataset README names this exact paper and the six TSV locations match.'},
        'entries':entries,'canonical_catalog_modified':False,'native_mask_localization_established':False,
        'anatomical_review_accepted':False,'training_admission_established':False,'automatic_seed_placement_allowed':False,
        'display_notice':'Dataset / paper report; anatomical review pending. Patient-level reported site does not establish native-mask localization.'}


def verify(document,root=ROOT):
    need(document==expected(root),'Sidecar differs from exact six source facts, identities or unaccepted review scope')


def verify_primary():
    with urllib.request.urlopen(PAPER_URI,timeout=20) as response:body=response.read(2_000_001)
    need(len(body)<=2_000_000 and sha256(body).hexdigest()==PAPER_SHA,'Primary archival XML changed or oversized')
    table=ET.fromstring(body).find(".//table-wrap[@id='T1']");need(table is not None,'Primary Table 1 missing')
    selected={}
    for row in table.findall('.//tr'):
        cells=[' '.join(''.join(cell.itertext()).split()) for cell in row]
        if cells and cells[0] in {'PAT'+number for number in FACTS}:selected[cells[0]]=(cells[5],cells[4])
    need(selected=={'PAT'+number:fact for number,fact in FACTS.items()},'Primary selected participant facts differ')
    return {'primary_xml_sha256':PAPER_SHA,'selected_rows_verified':6,'image_or_bundle_payloads_opened':False}


def controls(document):
    verify(document)
    mutations=[lambda d:d['entries'].pop(),lambda d:d['entries'].append(deepcopy(d['entries'][0])),
        lambda d:d['entries'][0].update(patient_group='BTC:sub-PAT26'),lambda d:d['entries'][0].update(role='SELECT'),
        lambda d:d['entries'][0].update(source_reported_lateralization='Right'),
        lambda d:d['entries'][0].update(hemisphere='Left'),lambda d:d['entries'][0].update(anatomical_review_accepted=True),
        lambda d:d['entries'][0].update(bundle_sha256='0'*64),lambda d:d['source'].update(sha256='0'*64),
        lambda d:d['paper_source'].update(sha256='0'*64),lambda d:d['catalog'].update(sha256='0'*64),
        lambda d:d.update(automatic_seed_placement_allowed=True)]
    for mutation in mutations:
        invalid=deepcopy(document);mutation(invalid)
        try:verify(invalid)
        except ValueError:pass
        else:raise AssertionError('Altered source evidence was admitted')
    return {'positive_controls':1,'negative_controls':len(mutations),'all_passed':True}


if __name__=='__main__':
    need(sys.argv[1:] in [[],['--verify-primary']],'Only optional --verify-primary is supported')
    document=json.loads((HERE/'evidence.json').read_text());result={'metadata_controls':controls(document)}
    if '--verify-primary' in sys.argv:result['primary_source']=verify_primary()
    print(json.dumps(result,indent=2))
