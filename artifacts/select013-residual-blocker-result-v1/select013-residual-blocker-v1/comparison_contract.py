"""One prospective native catalog at the predeclared old47-action frontier."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
H48 = ROOT/'build/select013-paged-route-h48-v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(H48/'comparison_contract.py') != '22ffd92bb134d1a7548f38a8e833427ed3b9aa0af7d9ff2fef90e922dc673c7a':
    raise ValueError('Historical H48 contract changed')
spec = importlib.util.spec_from_file_location('residual_h48_contract', H48/'comparison_contract.py')
h48_contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(h48_contract)
prior_contract = h48_contract.prior_contract
read_pinned = h48_contract.read_pinned
PRIOR = h48_contract.PRIOR
OUTPUT = 'build/select013-residual-blocker-v1/attempt-01'
ARMS = ('authenticated_old_frontier', 'prospective_residual_catalog')
WORKER_SECONDS, PARENT_SECONDS = 210, 240
MEMORY_BYTES, OUTPUT_BYTES, SUPERVISION_BYTES = 3*1024**3, 16*1024**2, 8*1024**2
LIMITS = {'max_steps': 48, 'fixed_prefix_steps': 47, 'base_candidates': 120,
    'extra_candidates': 32, 'branch_cap': 3, 'native_previews': 10000,
    'native_commits_and_STOP': 60, 'sidecar_cells': 16, 'threads': 1,
    'checkpoint_loads': 0, 'model_forwards': 0, 'updates': 0}
# Source construction <=120, 48 old-state catalogs <=48*152, prospective base
# <=120, extras <=32, three branches <=3*(prep+120+replay)=366: <=7,934 previews.
# 47 original prefix commits +1 STOP +3 prep+3 native replays =54 <=60.
PINS = {**h48_contract.PINS,
    'h48_result': {'path': 'build/select013-paged-route-h48-v1/attempt-01/result.json', 'sha256': '8dcbf51e12fc4a1175f52422bd5e4b745ce6d35300fd589b8b4b23404d7bdcf2'},
    'h48_receipt': {'path': 'build/select013-paged-route-h48-v1/attempt-01.supervision/receipt.json', 'sha256': 'f4924629b70c0c6a1a6b3bdf19b7fa7953076021a1788dfac7b5e2c26cde8b44'},
    'h48_release': {'path': 'build/select013-paged-route-h48-v1/root-release.json', 'sha256': 'a957077b04785355b3fd1c8354743526e4e70ae6087de249e325386badad8e3e'},
    'h48_index': {'path': 'build/select013-paged-route-h48-v1/source-index.json', 'sha256': '23bf8d7bb299a05f7f24dfa5d372b6aec6ee4f0f47a88e1f007a8109d458526f'},
    'frontier_page': {'path': 'build/select013-paged-route-h48-v1/attempt-01/paged_obstruction_lazy_beam/pages/166410d0dc42e97ead56ab095c66cce813db189f668a12c1eff54c5b512b8247/page00.json', 'sha256': 'a58f8ed4792633a3486bad99a86ebb780e421538b76d7c546062592a90e61917'}}


def historical():
    saved = h48_contract.historical()
    result, receipt, release, index, page = [read_pinned(PINS[k]) for k in
        ('h48_result', 'h48_receipt', 'h48_release', 'h48_index', 'frontier_page')]
    if (not h48_contract.complete_result(result) or receipt['status'] != 'complete'
            or receipt['exit_code'] != 0 or receipt['cleanup_errors'] or receipt['final_owned_pids']
            or not receipt['worker_termination_confirmed']
            or receipt['result_sha256'] != PINS['h48_result']['sha256']
            or receipt['release_sha256'] != PINS['h48_release']['sha256']
            or release['source_index'] != PINS['h48_index']):
        raise ValueError('Exact clean completed H48 predecessor required')
    accounting = result['arms']['paged_obstruction_lazy_beam']['selector_accounting']
    layer = accounting['layers'][46]
    opening = next(r for r in layer['opening_depth_retention'] if r['reason'] == 'opening')
    prefix = tuple(opening['actions']); checkpoints = []
    for depth in range(1, 48):
        rows = [r for r in accounting['layers'][depth-1]['retained_prefix_diagnostics']['retained_nonterminal_prefixes']
                if tuple(r['actions']) == prefix[:depth]]
        if len(rows) != 1: raise ValueError('Exact consecutive fixed opening prefix missing')
        checkpoints.append(rows[0])
    if (len(prefix) != 47 or layer['depth'] != 47 or not layer['completed']
            or page['steps_taken'] != 47 or page['page_index'] != 0
            or page['base_emitted_count'] != 117 or checkpoints[-1]['legal_nonstop_actions'] != 7
            or opening['estimated_incremental_return'] >= 0):
        raise ValueError('Declared old47 frontier differs')
    saved.update(h48_index=index, h48_result=result, prefix=prefix,
        prefix_checkpoints=checkpoints, opening=opening, frontier_page=page)
    return saved


def complete_result(result):
    branches = result.get('branch_count', -100)
    return (result.get('status') == 'complete_prospective_residual_frontier_diagnostic'
        and result.get('world_equal') is True and result.get('context_equal') is True
        and result.get('prefix_exact') is True and result.get('frontier_page_exact') is True
        and result.get('original_task_state_unchanged') is True and result.get('native_source_unchanged') is True
        and result.get('all_extra_pairs_verified') is True and result.get('original_negative_prefix_closed') is True
        and result.get('accepted_plan_claim') is False and result.get('new_rule_generated_prefix') is False
        and 0 <= branches <= LIMITS['branch_cap']
        and result.get('native_commits') == 47+2*branches and result.get('STOP_calls') == 1
        and result.get('native_commit_attempts') == result.get('native_commits')
        and result.get('independent_geometry_checks') == 1+branches
        and result.get('independent_geometry_all_feasible') is True
        and result.get('checkpoint_loads') == result.get('model_forwards') == result.get('updates') == 0
        and result.get('optimizer_attempts') == result.get('gradient_attempts') == 0
        and result.get('held_inputs_opened') is False and result.get('EVAL_opened') is False
        and result.get('private_reference_reads') == 0)


def source_guard(release, path, digest):
    if (sha(path) != digest or release.get('status') != 'released_one_attempt'
            or release.get('output') != OUTPUT or release.get('limits') != LIMITS
            or release.get('worker_seconds') != WORKER_SECONDS or release.get('parent_seconds') != PARENT_SECONDS
            or release.get('attempts') != 1 or release.get('automatic_retry') is not False
            or release.get('prior_pins') != PINS):
        raise ValueError('Exact single prospective diagnostic release required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip()
    index = read_pinned(release['source_index'])
    if head != release['expected_head'] or index['head'] != head: raise ValueError('HEAD changed')
    for section in ('source_files', 'metadata_files'):
        for name, digest in index[section].items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts or (ROOT/rel).is_symlink() or sha(ROOT/rel) != digest:
                raise ValueError('Bound file changed: '+name)
    historical()
    return index
