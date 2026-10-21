#!/usr/bin/env python3
"""Offline V8 log windowing. Never attaches to or controls an app.

Capture log byte lengths at two wall-clock markers, then (after graceful exit)
keep only complete tick records within those byte bounds. All non-tick records
remain byte-identical and in order, including code/library/move/delete events.
The selected window describes log writes, not an exact profiler-clock interval;
V8 output buffering can shift its relationship to the wall-clock markers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

SCHEMA = 'resectionlab.v8-byte-window.v1'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def capture(directory, pattern):
    rows = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            stat = path.stat()
            prefix = stream.read(min(4096, stat.st_size))
        rows[path.name] = dict(bytes=stat.st_size, device=stat.st_dev,
                               inode=stat.st_ino, prefixBytes=len(prefix), prefixSha256=digest(prefix))
    return dict(schema=SCHEMA, capturedUtc=datetime.now(timezone.utc).isoformat(),
                directory=str(directory.resolve()), pattern=pattern, files=rows)

def check_identity(path, row):
    stat = path.stat()
    if (stat.st_dev, stat.st_ino) != (row['device'], row['inode']) or stat.st_size < row['bytes']:
        raise ValueError(f'Captured log was replaced or truncated: {path}')
    with path.open('rb') as stream:
        if digest(stream.read(row['prefixBytes'])) != row['prefixSha256']:
            raise ValueError(f'Captured log prefix changed: {path}')

def filter_log(source, destination, begin, end):
    if not 0 <= begin <= end <= source.stat().st_size:
        raise ValueError('Invalid byte window')
    counts = dict(ticksKept=0, ticksExcluded=0, nonTickRecordsPreserved=0)
    times = []
    urls = set()
    source_before = source.stat()
    with source.open('rb') as stream, destination.open('xb') as output:
        offset = 0
        for line in stream:
            start, offset = offset, offset + len(line)
            if not line.endswith(b'\n'):
                raise ValueError('Incomplete final record; filter only after the profiled task exits gracefully')
            if line.startswith(b'tick,'):
                if start < begin or offset > end:
                    counts['ticksExcluded'] += 1
                    continue
                counts['ticksKept'] += 1
                try:
                    times.append(int(line.split(b',', 3)[2]))
                except (ValueError, IndexError):
                    pass
            else:
                counts['nonTickRecordsPreserved'] += 1
                if line.startswith(b'script-source,'):
                    fields = line.split(b',', 3)
                    if len(fields) > 2:
                        urls.add(fields[2].decode('utf-8', 'replace'))
            output.write(line)
    source_after = source.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
        raise ValueError('Source log changed during offline filtering')
    return dict(**counts, beginByte=begin, endByte=end,
                firstTickTimestampRaw=min(times) if times else None,
                lastTickTimestampRaw=max(times) if times else None, scriptUrls=sorted(urls))

def filter_captures(before, after, output_dir):
    if before['schema'] != SCHEMA or after['schema'] != SCHEMA:
        raise ValueError('Unknown capture schema')
    if (before['directory'], before['pattern']) != (after['directory'], after['pattern']):
        raise ValueError('Capture inventories differ')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, last in sorted(after['files'].items()):
        if Path(name).name != name:
            raise ValueError('Unsafe source basename')
        source = Path(after['directory']) / name
        first = before['files'].get(name)
        check_identity(source, last)
        if first:
            check_identity(source, first)
            if (first['device'], first['inode']) != (last['device'], last['inode']):
                raise ValueError('Log identity differs between markers')
        begin = first['bytes'] if first else 0
        destination = output_dir / (name + '.window.log')
        row = filter_log(source, destination, begin, last['bytes'])
        rows.append(dict(source=str(source), output=str(destination),
                         createdDuringWindow=first is None, **row))
    return dict(schema=SCHEMA, beforeUtc=before['capturedUtc'], afterUtc=after['capturedUtc'],
                caveat='Byte-write window; buffered profiler records may lag wall-clock markers.', files=rows)

def self_test(root):
    # Synthetic records only. No app, source case, engine or profiler is started.
    with tempfile.TemporaryDirectory(prefix='v8-window-selftest-', dir=root) as temporary:
        folder = Path(temporary)
        path = folder / 'isolate-tiny-v8-renderer.log'
        code = b'code-creation,JS,10,0,0x1000,8,tiny\n'
        old_tick = b'tick,0x1000,1,0,0,0\n'
        path.write_bytes(b'v8-version,13,0,0,0,0\n' + code + old_tick)
        before = capture(folder, '*v8-renderer.log')
        selected = b'tick,0x1000,2,0,0,0\n'
        with path.open('ab') as stream:
            stream.write(b'shared-library,/tiny/library,0,8\n' + selected)
        after = capture(folder, '*v8-renderer.log')
        later = b'code-move,0x1000,0x2000\n'
        with path.open('ab') as stream:
            stream.write(later + b'tick,0x2000,3,0,0,0\n')
        receipt = filter_captures(before, after, folder / 'windows')
        result = Path(receipt['files'][0]['output']).read_bytes()
        assert result == b'v8-version,13,0,0,0,0\n' + code + b'shared-library,/tiny/library,0,8\n' + selected + later
        assert receipt['files'][0]['ticksKept'] == 1
        # A partial record at either boundary must not admit that tick.
        partial = folder / 'partial.log'
        counts = filter_log(path, partial, before['files'][path.name]['bytes'] + 35,
                            after['files'][path.name]['bytes'] - 1)
        assert counts['ticksKept'] == 0
        print(json.dumps(dict(status='passed', syntheticOnly=True, checks=[
            'only complete in-window tick selected', 'all code/library/move records preserved byte-for-byte',
            'out-of-window ticks excluded', 'partial-boundary tick excluded', 'input logs unchanged'])))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    snap = commands.add_parser('capture')
    snap.add_argument('--directory', type=Path, required=True)
    snap.add_argument('--pattern', default='*v8-renderer.log')
    snap.add_argument('--output', type=Path, required=True)
    filt = commands.add_parser('filter')
    filt.add_argument('--before', type=Path, required=True)
    filt.add_argument('--after', type=Path, required=True)
    filt.add_argument('--output-dir', type=Path, required=True)
    check = commands.add_parser('self-test')
    check.add_argument('--temp-root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        write_json(args.output, capture(args.directory, args.pattern))
    elif args.command == 'filter':
        receipt = filter_captures(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.output_dir)
        write_json(args.output_dir / 'receipt.json', receipt)
        print(json.dumps(receipt, indent=2))
    else:
        self_test(args.temp_root)

if __name__ == '__main__':
    main()
