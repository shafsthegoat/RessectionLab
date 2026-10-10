"""One post-seal source-annotation comparison, with no policy or native search."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    release_path = Path(sys.argv[1])
    if digest(release_path) != sys.argv[2]:
        raise ValueError('Evaluation release changed')
    release = json.loads(release_path.read_text())
    output = ROOT / release['output']
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = {'status': 'started', 'private_load_calls': 0, 'private_array_reads': 0}
    try:
        for row in release['required_completed_files']:
            if digest(ROOT / row['path']) != row['sha256']:
                raise ValueError('Completed comparison evidence changed')
        parent = json.loads((ROOT / release['parent_receipt']).read_text())
        child = json.loads((ROOT / release['child_result']).read_text())
        if not (parent['status'] == 'complete' and parent['exit_code'] == 0
                and parent['worker_termination_confirmed'] is True
                and not parent['cleanup_errors'] and not parent['final_owned_pids']
                and child['status'] == 'complete_saved_plan_replay_comparison'):
            raise ValueError('Clean completed replay continuation required before reference access')
        import numpy as np
        from resectionlab.patient_planning_evaluation import evaluate_sealed_annotation_plans

        def load_reference():
            result['private_load_calls'] += 1
            arrays = {}
            for key in ('mask', 'coverage'):
                row = release['reference'][key]
                path = ROOT / row['path']
                if digest(path) != row['sha256']:
                    raise ValueError('Qualified annotation or coverage changed')
                value = np.load(path, allow_pickle=False)
                result['private_array_reads'] += 1
                if value.dtype != np.uint8 or not np.isin(value, (0, 1)).all():
                    raise ValueError('Binary saved reference required')
                arrays[key] = value.astype(bool)
            return {**arrays, **{key: release['reference'][key] for key in
                ('patient_group', 'public_source_hash', 'frame_correspondence',
                 'source_kind', 'label_name', 'source_sha256', 'affine_ras_mm')}}

        comparison = evaluate_sealed_annotation_plans(ROOT / release['bundle']['path'],
            expected_sha256=release['bundle']['sha256'], load_reference=load_reference)
        with (output / 'annotation-comparison.json').open('x') as stream:
            json.dump(comparison, stream, indent=2, allow_nan=False)
            stream.write('\n')
        result.update(status='complete', comparison_sha256=digest(output / 'annotation-comparison.json'),
                      model_calls=0, optimizer_updates=0, native_search_calls=0,
                      reference_scope='automatic source BrainLab annotation; not manual truth or injury evidence')
    except BaseException as error:
        result.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
