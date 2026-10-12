"""Bounded informed beam on the qualified local task, without teacher paths.

One paid complete catalog per expanded parent. Every legal row is ranked, but
only the stable union of two local-reward and two current-blocker choices is
committed. Live width two is separate from the completed STOP/success incumbent.
"""
import copy
import time
from dataclasses import dataclass

from local_goal import LocalEpisode, local_reward, reused
from resectionlab.core import semantic_digest


class SearchLimit(RuntimeError):
    pass


def branch(parent, index):
    """Clone owned certificates, never transplant a packet to an unowned state."""
    catalog = parent.observe()
    if type(index) is not int or not 0 <= index < len(catalog.identifiers):
        raise ValueError('Current certified union row required')
    child = copy.copy(parent)
    child.work = parent.work.clone()
    child.packets = list(parent.packets)
    child.rows = list(parent.rows)
    child.costs = dict(parent.costs)
    child.last_clean_snapshot = None
    pages = list(catalog.pages)
    pages[catalog.owners[index]] = pages[catalog.owners[index]].clone()
    child.catalog = reused.Catalog(catalog.context, child.work, pages,
        catalog.identifiers, catalog.owners, catalog.packets, catalog.scores,
        catalog.observation, catalog.record)
    child.step(index)
    return child


def propose(rows):
    """All finite legal rows eligible; stable original-inventory ties."""
    import math
    if any(not math.isfinite(r['local_reward']) or r['blocker_overlap'] < 0 for r in rows):
        raise ValueError('Finite certified local scores required')
    cost = sorted(rows, key=lambda r: (-r['local_reward'], r['index']))[:2]
    progress = sorted(rows, key=lambda r: (-r['blocker_overlap'], -r['local_reward'], r['index']))[:2]
    return tuple(dict.fromkeys(r['index'] for r in [*cost, *progress]))


@dataclass
class Node:
    episode: object
    path: tuple = ()
    seen: frozenset = frozenset()
    cleared: frozenset = frozenset()


def retain(nodes):
    """Terminal STOP never occupies either live lane."""
    live = [n for n in nodes if not n.episode.done]
    if not live:
        return []
    first = min(live, key=lambda n: (-n.episode.total_reward, n.path))
    others = [n for n in live if n.path != first.path]
    if not others:
        return [first]
    second = min(others, key=lambda n: (-len(n.cleared), -n.episode.total_reward, n.path))
    return [first, second]


def current_blockers(catalog):
    accounting = catalog.work._page_catalog.obstruction_accounting
    focus = None if accounting is None else accounting.get('goal_focus')
    if focus is None:
        return frozenset(), {'present': False, 'complete_first_failure_set': None}
    # The provider and Catalog have already authenticated this exact state,
    # source, fixed physical goal and all result certificates.
    if semantic_digest(focus['record']) != semantic_digest(catalog.work._goal_focus):
        raise ValueError('Current goal-focus evidence mismatch')
    cells = frozenset(tuple(int(v) for v in cell) for cell in focus['retained_cells'])
    first = focus['first_failure']
    return cells, {'present': True, 'cells': sorted(cells),
        'complete_first_failure_set': None if first is None else first['complete_first_failure_set'],
        'evidence': first, 'scope': 'bounded current first obstruction; not complete corridor or action lower bound'}


def search(context, snapshot, *, limits, usage, guard=lambda: None, progress=lambda **kw: None):
    started = time.perf_counter()
    origin = dict(usage())
    account = {'schema': 'train045-local-goal-beam-v1', 'complete': False,
        'width': 2, 'child_proposal_cap': 4, 'all_legal_rows_eligible': True,
        'global_optimality_proven': False, 'teacher_path_used': False,
        'positive_reward_guard': False, 'negative_preparation_allowed': True,
        'retention': 'one actual-return lane plus distinct known-goal-blocker-progress lane',
        'proposal_rule': 'stable union top2 immediate local reward and top2 current first-blocker overlap',
        'catalogs': 0, 'ranked_rows': 0, 'proposed_children': 0, 'evaluated_children': 0,
        'successful_children': 0, 'negative_children': 0, 'layers': [], 'cap_reason': None}
    def check(previews=0, transitions=0):
        guard()
        now = usage()
        if time.perf_counter()-started >= limits['seconds']:
            raise SearchLimit('search wall allowance; replay reserve preserved')
        if now['previews']-origin['previews']+previews > limits['native_previews']:
            raise SearchLimit('search preview allowance; replay reserve preserved')
        if now['transitions']-origin['transitions']+transitions > limits['native_transitions']:
            raise SearchLimit('search transition allowance; replay reserve preserved')
    root = Node(LocalEpisode(context, snapshot, guard=check))
    incumbent = None
    frontier = [root]
    try:
        check(previews=152, transitions=1)
        root.episode.observe()
        incumbent = branch(root.episode, 0)  # genuine canonical STOP, return zero
        if not incumbent.done or incumbent.total_reward != 0:
            raise ValueError('Exact root STOP incumbent required')
        while frontier:
            layer = {'depth': len(frontier[0].path), 'parents': [], 'complete': False}
            account['layers'].append(layer)
            children = []
            for node in frontier:
                check(previews=0 if node.episode.catalog is not None else 152)
                cat = node.episode.observe()
                blockers, evidence = current_blockers(cat)
                seen = node.seen | blockers
                rows = []
                for i in range(1, len(cat.identifiers)):
                    score = cat.scores[i]
                    value, will_succeed, _ = local_reward(context.goal, score,
                        node.episode.work._current_tool, already_succeeded=False)
                    removed = frozenset(tuple(c) for c in score['removed_indices_native'])
                    rows.append({'index': i, 'action_id': cat.identifiers[i], 'local_reward': value,
                        'goal_removal_predicted': will_succeed, 'blocker_overlap': len(removed & blockers),
                        'removed_indices_hash': semantic_digest(sorted(removed)),
                        'known_blockers_removed': sorted(removed & seen)})
                selected = propose(rows)
                parent = {'path': node.path, 'absolute_step': node.episode.work._steps,
                    'catalog_digest': cat._seal, 'observation_hash': cat.observation.fingerprint,
                    'source_state_hash': cat.work._engine.state_hash, 'blocker_evidence': evidence,
                    'rows': rows, 'selected_indices': selected, 'children': [], 'complete': False}
                layer['parents'].append(parent)
                account['catalogs'] += 1
                account['ranked_rows'] += len(rows)
                account['proposed_children'] += len(selected)
                for index in selected:
                    # A goal hit can perform one additional canonical STOP and
                    # its base inventory. Reserve that real work before commit.
                    check(previews=120, transitions=2)
                    child_episode = branch(node.episode, index)
                    candidate = rows[index-1]
                    if abs(child_episode.rows[-1]['reward']-candidate['local_reward']) > 1e-9:
                        raise ValueError('Certified branch score differs from committed local reward')
                    path = (*node.path, cat.identifiers[index])
                    cleared = node.cleared | frozenset(tuple(c) for c in candidate['known_blockers_removed'])
                    child = Node(child_episode, path, seen, cleared)
                    row = {'path': path, 'packet': child_episode.packets[len(node.episode.packets)],
                        'local_return': child_episode.total_reward, 'success': child_episode.success,
                        'done': child_episode.done, 'cleared_known_blockers': len(cleared),
                        'result_state_hash': child_episode.work._engine.state_hash,
                        'history_hash': semantic_digest(child_episode.work._history)}
                    parent['children'].append(row)
                    account['evaluated_children'] += 1
                    account['negative_children'] += int(child_episode.total_reward < 0)
                    account['successful_children'] += int(child_episode.success)
                    if child_episode.done:
                        if child_episode.total_reward > incumbent.total_reward:
                            incumbent = child_episode
                    else:
                        children.append(child)
                        # Streaming retention has the same two lane extrema;
                        # at most width2 live children plus the candidate exist.
                        children = retain(children)
                parent['complete'] = True
            frontier = retain(children)
            layer['retained_paths'] = [n.path for n in frontier]
            layer['complete'] = True
            progress(depth=layer['depth'], expanded=account['catalogs'],
                evaluated_children=account['evaluated_children'], incumbent=incumbent.total_reward,
                success=incumbent.success)
        account['complete'] = True
    except SearchLimit as error:
        if incumbent is None:
            raise
        account['cap_reason'] = str(error)
    except BaseException as error:
        account['seconds'] = time.perf_counter()-started
        account['native_usage_delta'] = {k: usage()[k]-origin[k] for k in origin}
        error.local_search_accounting = account
        raise
    account['seconds'] = time.perf_counter()-started
    account['native_usage_delta'] = {k: usage()[k]-origin[k] for k in origin}
    account['selected_return'] = incumbent.total_reward
    account['selected_success'] = incumbent.success
    account['selected_suffix_actions'] = len(incumbent.packets)
    account['completion_scope'] = ('declared pruned width2 search exhausted to horizon/terminal'
        if account['complete'] else 'incomplete declared search; selected incumbent is separately complete')
    return incumbent, account
