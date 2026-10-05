#!/usr/bin/env python3
"""Preserve raw episodes; create lossless gzip and verify gzip-only reporting."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
NAMES=('greedy','selection-1','selection-2','untrained-selection-1','untrained-selection-2')
HASH=lambda data:hashlib.sha256(data).hexdigest()

def main():
    records=[]
    for name in NAMES:
        path=ROOT/'profile'/name/'episode.json'
        raw=path.read_bytes()
        packed=gzip.compress(raw,compresslevel=9,mtime=0)
        unpacked=gzip.decompress(packed)
        assert unpacked==raw and json.loads(unpacked)==json.loads(raw)
        destination=path.with_suffix('.json.gz')
        destination.write_bytes(packed)
        assert path.read_bytes()==raw
        records.append({'raw_path':str(path.relative_to(ROOT)),'gzip_path':str(destination.relative_to(ROOT)),
            'uncompressed_sha256':HASH(raw),'gzip_sha256':HASH(packed),
            'raw_bytes':len(raw),'gzip_bytes':len(packed),'byte_roundtrip':True,'json_semantic_roundtrip':True,
            'raw_file_retained_unchanged':True})
    inputs=json.loads((ROOT/'report-source.json').read_text())['inputs']
    baseline=ROOT.parents[2]/'artifacts/validation/native-axis-public-v2/execution-baseline.json'
    with tempfile.TemporaryDirectory(prefix='axis-gzip-only-report-') as temporary:
        copy=Path(temporary)/'inputs';copy.mkdir()
        compressed_paths={record['raw_path']:record['gzip_path'] for record in records}
        for relative in inputs:
            source=ROOT/compressed_paths.get(relative,relative)
            target=copy/compressed_paths.get(relative,relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        report=Path(temporary)/'report.py';shutil.copyfile(ROOT/'report.py',report)
        output=Path(temporary)/'derived'
        completed=subprocess.run([sys.executable,str(report),'--inputs',str(copy),'--output',str(output),
            '--baseline',str(baseline)],capture_output=True,text=True,check=True)
        reproduced={}
        for filename in ('RESULT.md','cost-summary.json','report-source.json'):
            original=(ROOT/filename).read_bytes();actual=(output/filename).read_bytes()
            assert original==actual,filename
            reproduced[filename]={'sha256':HASH(original),'byte_identical':True}
        assert all(not (copy/record['raw_path']).exists() for record in records)
    receipt={'status':'passed','method':'gzip level9 mtime0; original raw files retained; byte and JSON roundtrip; report rerun with only compressed episode files',
        'files':records,'raw_bytes_total':sum(r['raw_bytes'] for r in records),
        'gzip_bytes_total':sum(r['gzip_bytes'] for r in records),'gzip_only_report_reproduction':reproduced,
        'report_script_sha256':HASH((ROOT/'report.py').read_bytes()),'packer_script_sha256':HASH(Path(__file__).read_bytes()),
        'new_episodes':0,'gradient_steps':0,'original_receipts_changed':False}
    (ROOT/'compression-roundtrip.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':receipt['status'],'raw_bytes':receipt['raw_bytes_total'],'gzip_bytes':receipt['gzip_bytes_total'],
        'gzip_only_report_reproduction':True},indent=2))

if __name__=='__main__':
    main()
