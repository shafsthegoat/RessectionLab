#!/usr/bin/env python3
"""Package a separate real CT/MRA case using an already saved inference result."""
from pathlib import Path
import argparse
import json
from resectionlab.patient_workspace import prepare_model_workspace


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference',type=Path,required=True)
    parser.add_argument('--source-metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();source=json.loads(args.source_metadata.read_text())
    output=args.output or root/'outputs/poc-cases'/f"{source['case_id']}.ressectionlab"
    source['metadata_paths']=[str(args.source_metadata.resolve()),*[str((root/path).resolve()) for path in source.get('metadata_paths',[])]]
    print(json.dumps(prepare_model_workspace(args.inference,output,source_metadata=source),indent=2))


if __name__=='__main__':main()
