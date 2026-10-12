"""Lossless completed-plan persistence; no scores, choices or physical semantics."""
from common import semantic,write_json,sha
from comparison import plain
from compact_output import encode_gzip


def persist_completed_rollout(output,sink,report,method,packets,account):
    """Persist expensive returned decisions BEFORE subsequent validation can fail."""
    packets,account=plain(packets),plain(account)
    raw,binding=encode_gzip(account);path=method+'-selection.json.gz'
    sink.write_bytes(output/path,raw)
    report['selections'][method]={'path':path,**binding,'validation':'pending',
        'complete':account.get('complete') is True,'packet_count':len(packets),
        'model_transition_calls':account.get('model_transition_calls'),
        'planned_total_return':sum(r['reward'] for r in account['explored_history']),
        'planned_suffix_return':sum(r['reward'] for r in account['explored_history'][account['fixed_prefix_length']:])}
    write_json(output/'report.json',plain(report))
    return packets,account


def seal_plan_pair(output,sink,report,selected,accounts,frontier,*,native_transition_attempts):
    selected,accounts,frontier=plain(selected),plain(accounts),plain(frontier)
    if set(selected)!={'GREEDY','TAIL'} or set(accounts)!={'GREEDY','TAIL'}:
        raise ValueError('Both full method plans required')
    if any(accounts[m].get('complete') is not True or report['selections'][m]['validation']!='accepted'
            for m in ('GREEDY','TAIL')):
        raise ValueError('Both fully validated plans required before seal')
    sealed={m:{'selected_action_records':selected[m],
        'planned_history_hash':semantic(accounts[m]['explored_history']),
        'planned_total_return':sum(r['reward'] for r in accounts[m]['explored_history']),
        'frontier_seal':frontier['seal']} for m in ('GREEDY','TAIL')}
    sealed={k:{**v,'seal':semantic(v)} for k,v in sealed.items()}
    sink.write(output/'plans.json',sealed)
    report['plans_sealed_before_replay']={'sha256':sha(output/'plans.json'),
        'seals':{k:v['seal'] for k,v in sealed.items()},'native_transition_attempts_at_seal':native_transition_attempts}
    write_json(output/'report.json',plain(report))
    return sealed
