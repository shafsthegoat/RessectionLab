#!/usr/bin/env python3
"""Qualify existing named bundles without downloads, inference or role changes."""
from pathlib import Path
import argparse
import json
from resectionlab.patient_corpus import default_plan, query_catalog, run_qualification


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=root)
    parser.add_argument('--output',type=Path,default=root/'outputs/patient-corpus-v1')
    parser.add_argument('--plan-only',action='store_true',help='Print metadata-only nine-person allowlist; do not open any imaging/bundle')
    parser.add_argument('--limit',type=int,help='Process at most this many people for a resumable partial run')
    parser.add_argument('--query',help='Search existing catalog only; no image access')
    parser.add_argument('--anatomy',choices=['brain','tumor','skull','vessels','ventricles'])
    parser.add_argument('--modality')
    parser.add_argument('--role',choices=['TRAIN','DEVELOPMENT'])
    args=parser.parse_args()
    if args.limit is not None and not 1<=args.limit<=9:parser.error('--limit must be 1 through 9')
    if args.query is not None or args.anatomy or args.modality or args.role:
        result=query_catalog(args.output.resolve(),text=args.query or '',anatomy=args.anatomy,modality=args.modality,role=args.role)
    else:
        plan=default_plan(args.root)
        result=plan if args.plan_only else run_qualification(args.root,args.output,plan,limit=args.limit)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
