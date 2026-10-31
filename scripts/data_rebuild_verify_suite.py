#!/usr/bin/env python3
"""Serial actual hidden desktop checks on the curated data core."""
import json
import os
from pathlib import Path
import subprocess

from resectionlab.data_root import atomic_json,checked_root,sha256_file

REPO=Path(__file__).resolve().parents[1]
CASES=[
 ('healthy-ixi016-control','controls/IXI-016-control-native-v5.ressectionlab',True),
 ('real-ucsf0040-source-limit','real/UCSF-PDGM-0040-structural-v2.ressectionlab',False),
 ('real-ucsf0061-outlier','real/UCSF-PDGM-0061-structural-v1.ressectionlab',False),
 ('real-ucsf0005-existing','real/UCSF-PDGM-0005-existing-v2.ressectionlab',True),
 ('measured-topbrain001-cta','clinical/TopCoW-001-CTA-v3.ressectionlab',False),
 ('measured-topbrain001-mra','clinical/TopCoW-001-TOF-MRA-v3.ressectionlab',False),
 ('synthetic-ixi016-geometry','synthetic/IXI-016-SYNTHETIC-geometry-00-v5.ressectionlab',True),
]


def run(root=None):
    root=checked_root(root,required_bytes=3*1024**3)
    document={'schema':'ressectionlab-curated-hidden-suite-v1','cases':[]}
    index=root/'qc/hidden-app/curated-suite-v1.json'
    for name,relative,rehearse in CASES:
        source=root/'staging/candidates'/relative
        if not source.is_file():
            document['cases'].append({'name':name,'status':'source_not_ready','path':str(source)});atomic_json(index,document);continue
        source_sha=sha256_file(source)
        attempts=1
        while True:
            output=root/'qc/hidden-app'/(name+f'-settled-v{attempts}')
            report=output/'report.json'
            if report.exists():
                previous=json.loads(report.read_text())
                if previous.get('passed') and previous.get('inputSha256')==source_sha:break
            if not output.exists():break
            attempts+=1
        if not report.exists():
            checked_root(root,required_bytes=1024**3)
            log=output.with_suffix('.launch.log')
            environment={**os.environ,'RESSECTIONLAB_DATA_ROOT':str(root),'DATA_REBUILD_CASE':str(source),
                         'DATA_REBUILD_OUTPUT':str(output),'DATA_REBUILD_REHEARSE':'1' if rehearse else '0'}
            with log.open('x') as stream:
                result=subprocess.run(['node','node_modules/electron/cli.js','electron/verify-data-rebuild.cjs'],
                    cwd=REPO/'desktop',env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=660)
            if not report.exists():
                document['cases'].append({'name':name,'status':'startup_failed','log':str(log.relative_to(root)),
                    'returncode':result.returncode});atomic_json(index,document);continue
        proof=json.loads(report.read_text())
        document['cases'].append({'name':name,'path':str(source.relative_to(root)),'sha256':source_sha,
            'status':'passed' if proof.get('passed') else 'failed','report':str(report.relative_to(root)),
            'report_sha256':sha256_file(report),'window_ever_visible':proof.get('windowEverVisible')})
        atomic_json(index,document);print(json.dumps(document['cases'][-1]),flush=True)
    return document


if __name__=='__main__':run()
