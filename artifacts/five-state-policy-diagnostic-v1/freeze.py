"""Root-only metadata freeze after canonical integration; no model/patient loads."""
import subprocess
from pathlib import Path
from diagnostic_worker import ROOT, HERE, OUTPUT, TRAIN, RUNS, CAPS, read, sha, completed_training


def main():
    import json
    prior = completed_training()
    original = read(prior[8]['root']/'source-index.json')
    sources = set(original['source_files'])
    sources.update(str(p.relative_to(ROOT)) for p in (HERE/'diagnostic_worker.py', HERE/'run_owned.py', HERE/'freeze.py'))
    # The prior closure covers the canonical factory/model/native replay. Include
    # current SELECT admission dependency if its backwards-compatible branch is present.
    for name in ('patient_select_inference.py',):
        path = ROOT/'src/resectionlab'/name
        if path.is_file(): sources.add(str(path.relative_to(ROOT)))
    metadata = set(original['metadata_files'])
    for updates, record in prior.items():
        parent = record['root']
        metadata.update(str((parent/p).relative_to(ROOT)) for p in (
            'root-release.json','source-index.json','attempt-01/result.json',
            'attempt-01.supervision/receipt.json','attempt-01/checkpoint-freeze.json',
            'attempt-01/teacher-readiness.json'))
        for subject in TRAIN:
            metadata.add(str((parent/('attempt-01/'+subject+'-context.json')).relative_to(ROOT)))
            metadata.add(str((parent/('attempt-01/teachers/'+subject+'/complete-trace.json')).relative_to(ROOT)))
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    index = dict(head=head, source_files={p:sha(ROOT/p) for p in sorted(sources)},
        metadata_files={p:sha(ROOT/p) for p in sorted(metadata)},
        checkpoint_authority='completed saved result plus clean owned parent; bounded loader verifies weights at actual run',
        public_arrays='only fixed four public manifests through canonical factory; no array reads during freeze')
    def write(path, value):
        with path.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
    target = HERE/'source-index.json'; write(target,index)
    release = dict(status='awaiting_root_release', expected_head=head, output=OUTPUT, TRAIN=list(TRAIN),
        caps=CAPS, attempts=1, automatic_retry=False, SELECT_EVAL_execution=False, private_reference_execution=False,
        source_index=dict(path=str(target.relative_to(ROOT)),sha256=sha(target)),
        readouts=['INITIAL','IL_1','RL_1','IL_8','RL_8'], teacher_states=5,
        baseline='NativeSpatialTask.observed_greedy_search through unchanged horizon24; complete replay',
        historical_training='separate completed 1/8 runs, no optimizer state or updates admitted here')
    write(HERE/'release-template.json',release)
    print(json.dumps(dict(source_index_sha256=sha(target),template_sha256=sha(HERE/'release-template.json'))))


if __name__ == '__main__': main()
