"""Exact existing TRAIN045 source/record pins; no weights or cached payload input."""
import paths
from common import ROOT,IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,read_json
PRIOR_INDEX='build/train045-focused-batched-v1/runner/source-index.json'
PRIOR_INDEX_SHA='8bf1c3def8f773f5b9b4505392b995886d8cba87a96e3eb91af926d23ccb4df6'
PREFIX_REPORT='build/continuation-endgame-pair-v1/recovery-01/report.json'
PREFIX_REPORT_SHA='e0ab3670571e64ddf5818c190a7d599f55f96dd5e59be24ca29b1038e7b29751'
PREFIX_PLANS='build/continuation-endgame-pair-v1/run-01/plans.json'
PREFIX_PLANS_SHA='6798288cbb05cce8692f61f64438b68fcad8aef15ebc5ec008b2f3e3a151521b'
REFERENCE_ROOT='build/train045-frontier-preparation-v1/recovery-compact-01/'
REFERENCE_REPORT=REFERENCE_ROOT+'report.json'
REFERENCE_PLANS=REFERENCE_ROOT+'plans.json'
REFERENCE_FRONTIER=REFERENCE_ROOT+'original-frontier.json'
PINS={PREFIX_REPORT:PREFIX_REPORT_SHA,PREFIX_PLANS:PREFIX_PLANS_SHA,
    REFERENCE_REPORT:'3d9898b3222edca89ec681ed1a6cba7dd62c54d779638e98285341ef3258d35d',
    REFERENCE_PLANS:'2c33637e7c9f6c6609910009a4e7d6f4b9a1a8b9764a905cbf156fbea2df24ec',
    REFERENCE_FRONTIER:'039b09fbe7b87160738c9577645c24009b422b2daf463e249016c3bcf9640c76',
    IL+'/configuration.json':'55f11d31f34f5f44d697d25f010b1a6acd1a462533cf4ecf1010c36e8af1713b',
    PUBLIC:PUBLIC_SHA,COHORT:COHORT_SHA,PRIOR_INDEX:PRIOR_INDEX_SHA}

def metadata_pins():
    prior=read_json(ROOT/PRIOR_INDEX,PRIOR_INDEX_SHA);pins=dict(PINS)
    public=read_json(ROOT/PUBLIC,PUBLIC_SHA)
    for row in public['cases']:
        from pathlib import Path
        name=str(Path(row['path']).relative_to(ROOT));pins[name]=row['sha256']
    arrays=prior['public_arrays']
    if len(arrays)!=5:raise ValueError('Exactly inherited five TRAIN045 source arrays required')
    return pins,arrays
