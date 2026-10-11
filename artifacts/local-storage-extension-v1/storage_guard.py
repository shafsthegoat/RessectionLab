#!/usr/bin/env python3
"""Owner-operated, advisory storage reservations; never launches/stops/deletes work."""
from __future__ import annotations
import argparse, datetime, fcntl, hashlib, json, os, re, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'build/storage-extension-v1'
MIB = 1024**2
CAP = 10 * 1024**3
FREE_FLOOR = 50 * 1024**3
OWNER = '/root/acquisition_continuity_audit'
OPEN = {'prepared', 'running'}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def atomic(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if len(data) > MIB:
        raise ValueError('ledger exceeds 1 MiB cap')
    fd, name = tempfile.mkstemp(prefix='.ledger-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)  # own failed atomic-write temp only; never user/project data

def safe_scope(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('scope must be a relative project path')
    if not (ROOT / p).resolve().is_relative_to(ROOT):
        raise ValueError('scope escapes project through a symlink')
    return p.as_posix()

def du_scope(value):
    p = ROOT / safe_scope(value)
    if not p.exists(): return 0
    result = subprocess.run(['/usr/bin/du', '-sk', str(p)], text=True,
                            capture_output=True, timeout=60, check=True)
    if result.stderr: raise ValueError(result.stderr.strip())
    return int(result.stdout.split()[0]) * 1024

def measure():
    started = now()
    result = subprocess.run(['/usr/bin/du', '-k', '-d', '1', '.'], cwd=ROOT,
                            text=True, capture_output=True, timeout=120, check=True)
    if result.stderr: raise ValueError(result.stderr.strip())
    entries = {}; total = None
    for line in result.stdout.splitlines():
        number, path = line.split('\t', 1)
        size = int(number) * 1024
        if path == '.': total = size
        elif path.startswith('./'): entries[path[2:]] = size
        else: raise ValueError('unexpected du path')
    if total is None: raise ValueError('du omitted project total')
    entries['@root-files-and-directory'] = total - sum(entries.values())
    if entries['@root-files-and-directory'] < 0:
        raise ValueError('inconsistent du totals')
    stat = os.statvfs(ROOT)
    return {'started_at_utc': started, 'ended_at_utc': now(),
            'allocated_bytes': entries, 'total_allocated_bytes': total,
            'free_bytes': stat.f_bavail * stat.f_frsize,
            'method': 'du -k -d 1 .; allocated file blocks, no bodies or symlink traversal; statvfs f_bavail'}

def update_measurement(ledger, baseline, measurement):
    high = ledger.setdefault('scope_highwater_bytes', dict(baseline['allocated_bytes']))
    for key, value in measurement['allocated_bytes'].items():
        high[key] = max(high.get(key, 0), value)
    ledger['growth_highwater_bytes'] = sum(max(0, value - baseline['allocated_bytes'].get(key, 0))
                                          for key, value in high.items())
    ledger['latest_measurement'] = measurement
    ledger['free_space_check'] = {'at_utc': measurement.get('ended_at_utc', now()), 'free_bytes': measurement['free_bytes']}
    for item in ledger['reservations'].values():
        if item['status'] in OPEN:
            current = sum(du_scope(s) for s in item['scopes'])
            item['scope_current_allocated_bytes'] = current
            item['observed_scope_growth_highwater_bytes'] = max(
                item.get('observed_scope_growth_highwater_bytes', 0),
                current - item['scope_initial_allocated_bytes'])

def capacity(ledger):
    reserved = sum(r['reserved_bytes'] for r in ledger['reservations'].values() if r['status'] in OPEN)
    growth = ledger['growth_highwater_bytes']
    free = ledger.get('free_space_check', ledger['latest_measurement'])['free_bytes']
    over = [key for key, r in ledger['reservations'].items()
            if r['status'] in OPEN and r.get('observed_scope_growth_highwater_bytes', 0) > r['reserved_bytes']]
    return {'growth_highwater_bytes': growth, 'outstanding_reserved_bytes': reserved,
            'conservative_growth_plus_reservations_bytes': growth + reserved,
            'additional_reservable_bytes': max(0, min(CAP - growth - reserved, free - FREE_FLOOR - reserved)),
            'free_bytes': free, 'free_after_outstanding_reservations_bytes': free - reserved,
            'reservation_scope_overruns': over,
            'allocation_allowed': growth + reserved <= CAP and free - reserved >= FREE_FLOOR and not over}

def require_capacity(ledger):
    report = capacity(ledger)
    if not report['allocation_allowed']:
        raise ValueError('storage limit/refusal: ' + json.dumps(report))

def reserve(ledger, key, mib, owner, scopes, pilot):
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', key): raise ValueError('invalid id')
    if key in ledger['reservations']: raise ValueError('reservation id already exists; no silent reset')
    if mib <= 0: raise ValueError('positive integer MiB required')
    scopes = [safe_scope(s) for s in scopes]
    if len(scopes) != len(set(scopes)): raise ValueError('duplicate scope')
    all_paths = [(key, Path(s)) for s in scopes]
    all_paths += [(k, Path(s)) for k, r in ledger['reservations'].items()
                  if r['status'] in OPEN for s in r['scopes']]
    for i, (a_key, a) in enumerate(all_paths):
        for b_key, b in all_paths[i+1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError('overlapping reservation scopes: '+a_key+' / '+b_key)
    item = {'status': 'prepared', 'owner': owner, 'pilot': pilot,
            'reserved_bytes': mib * MIB, 'scopes': scopes, 'created_at_utc': now(),
            'scope_initial_allocated_bytes': sum(du_scope(s) for s in scopes),
            'observed_scope_growth_highwater_bytes': 0,
            'launch_authorized_by_storage_reservation': False}
    ledger['reservations'][key] = item
    try: require_capacity(ledger)
    except Exception:
        del ledger['reservations'][key]
        raise

def begin(ledger, key):
    r = ledger['reservations'][key]
    if r['status'] != 'prepared': raise ValueError('only prepared reservation may begin')
    if r['pilot'] and any(x['pilot'] and x['status'] == 'running' for x in ledger['reservations'].values()):
        raise ValueError('another new dataset pilot is running; one dataset pilot at a time')
    require_capacity(ledger)
    r['status'] = 'running'; r['began_at_utc'] = now()

def close(ledger, key, terminal):
    r = ledger['reservations'][key]
    if r['status'] not in OPEN: raise ValueError('reservation already closed')
    path = ROOT / safe_scope(terminal)
    if path.stat().st_size > 4 * MIB: raise ValueError('terminal receipt exceeds cap')
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get('remaining_owned_pids') != [] or receipt.get('worker_termination_confirmed') is not True:
        raise ValueError('receipt must affirm no remaining owned process and termination')
    r['status'] = 'closed'; r['closed_at_utc'] = now()
    r['terminal_receipt'] = {'path': terminal, 'sha256': hashlib.sha256(raw).hexdigest()}
    r['terminal_note'] = 'Storage release only; success/admission not inferred from exit status.'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['snapshot', 'reserve', 'begin', 'close'])
    p.add_argument('--id'); p.add_argument('--mib', type=int); p.add_argument('--owner')
    p.add_argument('--scope', action='append', default=[]); p.add_argument('--pilot', action='store_true', help='new dataset pilot; scientific compute is separately serialized by root')
    p.add_argument('--terminal-receipt')
    args = p.parse_args()
    with (HERE / '.ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        baseline = read(HERE / 'baseline.json')
        ledger = read(HERE / 'ledger.json')
        if baseline['root'] != str(ROOT):
            raise ValueError('host/project baseline cannot be reused in another checkout')
        if ledger['baseline_sha256'] != hashlib.sha256((HERE/'baseline.json').read_bytes()).hexdigest():
            raise ValueError('baseline changed')
        if args.action == 'begin':
            stat = os.statvfs(ROOT)
            ledger['free_space_check'] = {'at_utc': now(), 'free_bytes': stat.f_bavail * stat.f_frsize}
        else:
            update_measurement(ledger, baseline, measure())
        atomic(HERE/'ledger.json', ledger)  # retain actual measurement even if next action refuses
        if args.action == 'reserve':
            if not args.id or not args.mib or not args.owner or not args.scope: p.error('reserve needs id, mib, owner, scope')
            reserve(ledger, args.id, args.mib, args.owner, args.scope, args.pilot)
        elif args.action == 'begin':
            if not args.id: p.error('begin needs id')
            begin(ledger, args.id)
        elif args.action == 'close':
            if not args.id or not args.terminal_receipt: p.error('close needs id, terminal-receipt')
            close(ledger, args.id, args.terminal_receipt)
        ledger['last_owner_action_at_utc'] = now()
        atomic(HERE/'ledger.json', ledger)
        print(json.dumps({'action': args.action, 'id': args.id, **capacity(ledger)}, indent=2))

if __name__ == '__main__': main()
