#!/usr/bin/env python3
"""Qualify existing named bundles without downloads, inference or role changes."""
from pathlib import Path
import argparse
import json
from resectionlab.patient_corpus import DEFAULT_RELEASE, IXI_RELEASE, RELEASES, release_plan, query_catalog, run_qualification


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=root)
    parser.add_argument('--release',choices=RELEASES,default=DEFAULT_RELEASE,help='Select an explicitly frozen release; default remains nine people')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true',help='Print metadata-only fixed release allowlist; do not open any imaging/bundle')
    parser.add_argument('--limit',type=int,help='Process at most this many people for a resumable partial run')
    parser.add_argument('--query',help='Search existing catalog only; no image access')
    parser.add_argument('--anatomy',choices=['brain','tumor','skull','vessels','ventricles'])
    parser.add_argument('--modality')
    parser.add_argument('--role',choices=['TRAIN','DEVELOPMENT'])
    args=parser.parse_args()
    count=10 if args.release==IXI_RELEASE else 9
    if args.output is None:args.output=args.root/('outputs/patient-corpus-ixi265-v1' if args.release==IXI_RELEASE else 'outputs/patient-corpus-v1')
    if args.limit is not None and not 1<=args.limit<=count:parser.error(f'--limit must be 1 through {count}')
    if args.query is not None or args.anatomy or args.modality or args.role:
        result=query_catalog(args.output.resolve(),text=args.query or '',anatomy=args.anatomy,modality=args.modality,role=args.role)
    else:
        plan=release_plan(args.root,args.release)
        result=plan if args.plan_only else run_qualification(args.root,args.output,plan,limit=args.limit)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
