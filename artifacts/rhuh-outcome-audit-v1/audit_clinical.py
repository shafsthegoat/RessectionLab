"""Descriptive audit of the unchanged public TCIA table; no models or image IO."""
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'clinical_data_TCIA_RHUH-GBM.csv'
EXPECTED_SHA = '32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), delimiter=',')
rows = list(reader); fields = reader.fieldnames
assert len(fields) == len(set(fields)) == 23
assert all(None not in row and all(value is not None for value in row.values()) for row in rows)
ids = [row['Patient ID'] for row in rows]
assert len(ids) == len(set(ids)) == 40
assert ids == [f'RHUH-{i:04d}' for i in range(1, 41)]
missing_tokens = ('', 'na', 'n/a', 'nan', 'null', 'none', 'unknown', 'not available')
columns = {}
for field in fields:
    values = [row[field] for row in rows]
    count = Counter(values)
    empty = sum(value.strip() == '' for value in values)
    missing = sum(value.strip().casefold() in missing_tokens for value in values)
    record = {'empty_cells': empty, 'explicit_missing_token_cells': missing,
              'unique_values': len(count), 'surrounding_whitespace_cells': sum(value != value.strip() for value in values)}
    try:
        nums = list(map(float, values))
    except ValueError:
        if field != 'Patient ID':
            record['observed_values'] = dict(sorted(count.items()))
    else:
        record['numeric_summary'] = {'min': min(nums), 'max': max(nums), 'mean': statistics.mean(nums),
                                     'median': statistics.median(nums), 'sample_sd': statistics.stdev(nums)}
    columns[field] = record
outcome = 'Postoperative Neurological Deficit'
eor = 'Extent of resection [EOR]  %'
eor_category = next(field for field in fields if field.startswith('EOR ='))
pre_ce = 'Preoperative  contrast enhancing tumor volume (cm3)'
post_ce = 'Postoperative contrast enhancing residual tumor (cm3)'
deltas = [int(row['Postoperative KPS'])-int(row['Preoperative KPS']) for row in rows]
contradictions = []
for row in rows:
    value = float(row[eor]); computed = 100*(1-float(row[post_ce])/float(row[pre_ce]))
    if row[eor_category] == 'NTR' and not value > 95:
        contradictions.append({'patient_id':row['Patient ID'],'source_category':row[eor_category],
            'source_EOR_percent':value,'source_pre_ce_cm3':float(row[pre_ce]),'source_post_ce_cm3':float(row[post_ce]),
            'EOR_recomputed_from_rounded_source_volumes_percent':computed,
            'interpretation':('Recorded NTR label conflicts with >95% definition and recomputed rounded volumes.' if computed <= 95 else 'Displayed 95.0% is not strictly >95%; rounded source volumes imply >95%, so this is a boundary-rounding ambiguity, not a confirmed eligibility violation.') + ' No relabeling or exclusion applied.'})
crosstab = {}
for row, delta in zip(rows,deltas):
    status = 'decreased' if delta < 0 else 'increased' if delta > 0 else 'unchanged'
    crosstab.setdefault(row[outcome],Counter())[status] += 1
report = {'schema':'rhuh-public-clinical-audit-v1','source':{'filename':SOURCE.name,'sha256':EXPECTED_SHA,'bytes':len(raw)},
    'parser':{'encoding':'utf-8-sig','delimiter':',','header':True,'source_headers_preserved':True},
    'patient_count':len(rows),'column_count':len(fields),'patient_id_header':'Patient ID','patient_ids':ids,
    'duplicate_ids':0,'id_sequence_complete_0001_0040':True,'columns':columns,
    'outcome_counts':dict(sorted(Counter(row[outcome] for row in rows).items())),
    'derived_label_counts':{'any_recorded_category_other_than_No':sum(row[outcome]!='No' for row in rows),
        'recorded_persistent_categories':sum(row[outcome] in ('Minor Persistent','Major Persistent') for row in rows)},
    'KPS_change':{'definition':'Postoperative KPS minus Preoperative KPS; timing unknown; global function, not domain deficit',
       'counts':dict(sorted(Counter(deltas).items())),'decreased':sum(v<0 for v in deltas),
       'unchanged':sum(v==0 for v in deltas),'increased':sum(v>0 for v in deltas),
       'min':min(deltas),'max':max(deltas),'median':statistics.median(deltas),
       'by_recorded_deficit':{k:dict(v) for k,v in sorted(crosstab.items())}},
    'EOR_definition_conflicts':contradictions,
    'all_cell_count':len(rows)*len(fields),
    'empty_cell_count':sum(c['empty_cells'] for c in columns.values()),
    'explicit_missing_token_cell_count':sum(c['explicit_missing_token_cells'] for c in columns.values()),
    'structurally_absent_outcome_fields':['preoperative domain-specific neurological deficit','postoperative domain',
       'clinical assessment date or days since surgery','new or worsened relative to baseline','minor/major severity definition',
       'transient/persistent duration threshold','longitudinal recovery observations'],
    'fit_or_patient_image_processing':False,'source_values_changed':False}
(ROOT/'clinical-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['patient_count','column_count','outcome_counts','KPS_change','EOR_definition_conflicts','empty_cell_count','explicit_missing_token_cell_count']},indent=2))
