#!/usr/bin/env python3
"""Build/reopen the real UCSF-PDGM-0004 imaging workspace from pinned local sources."""
from pathlib import Path
import argparse
import json
from resectionlab.patient_workspace import prepare_ucsf_workspace


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=root/'manifests/data_acquisition_ucsf.json')
    parser.add_argument('--data-root',type=Path,default=root/'data')
    parser.add_argument('--output',type=Path,default=root/'outputs/poc-cases/UCSF-PDGM-0004.ressectionlab')
    parser.add_argument('--brain-mask',type=Path)
    parser.add_argument('--brain-inference',type=Path)
    parser.add_argument('--tumor-inference',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare_ucsf_workspace(args.manifest,args.data_root,args.output,
        brain_mask=args.brain_mask,brain_inference=args.brain_inference,tumor_inference=args.tumor_inference),indent=2))


if __name__=='__main__':main()
