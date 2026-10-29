#!/usr/bin/env python3
"""Encode sampled actual Electron captures on their recorded native time axis.

No geometry interpolation, crossfades or generated frames. The explicit playback
speed changes presentation time only. Raw frames and identity manifest remain
alongside the video, under ignored testcases/.
"""
from pathlib import Path
import argparse, json, re, subprocess

ROOT = Path(__file__).resolve().parents[1]


def encode(directory: Path, speed: float):
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / 'testcases'):
        raise ValueError('Use an existing evidence directory under testcases/')
    report = json.loads((directory / 'report.json').read_text())
    if not report.get('passed') or not report.get('observed'):
        raise ValueError('A passing actual replay receipt and observed captures are required')
    if not 0 < speed <= 16:
        raise ValueError('Presentation speed must be in (0,16]')
    output = directory / 'operative-rehearsal.mp4'
    if output.exists():
        raise ValueError('Preserve the prior recording')
    frames = report['observed']
    def native_time(frame):
        match = re.search(r'Native operative state · ([0-9.]+) s', frame['readout'])
        if not match:
            raise ValueError('Missing captured native time')
        return float(match.group(1))
    times = list(map(native_time, frames))
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Recording must be monotonic, without scrub-back')
    entries = [('01-before-field.png', 2.0)]
    for i, frame in enumerate(frames):
        duration = max(.04, ((times[i+1]-times[i])/speed if i+1 < len(times) else .5))
        if frame['id'] != frame['readoutState'] or frame['id'] != frame['frame']['state']:
            raise ValueError('Surface, HUD and recorded native state must agree')
        entries.append((frame['file'], duration))
    entries.append(('03b-after-tools-clear-field.png', 3.0))
    lines = []
    for name, duration in entries:
        if not re.fullmatch(r'(observed-\d+|01-before-field|03b-after-tools-clear-field)\.png', name) or not (directory / name).is_file():
            raise ValueError('Unexpected capture path')
        lines += [f"file '{name}'", "option framerate 1000", f'duration {duration:.6f}']
    lines += [f"file '{entries[-1][0]}'", "option framerate 1000", "duration 0.001"]
    (directory / 'recording.ffconcat').write_text('\n'.join(lines)+'\n')
    (directory / 'recording-caption.txt').write_text(
        f'Actual Electron native states | {speed:g}x native intervals | sampled GPU captures\n'
        'Illustrative materials | 1 mm source cells | authored research exposure\n')
    # Text is read from a file; never interpreted as shell code.
    filters = ("pad=ceil(iw/2)*2:ceil(ih/2)*2,"
               "drawtext=textfile=recording-caption.txt:fontsize=13:fontcolor=white:"
               "box=1:boxcolor=black@0.65:boxborderw=5:x=12:y=26,"
               "fps=30,format=yuv420p")
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n', '-f', 'concat',
                    '-safe', '0', '-i', 'recording.ffconcat', '-vf', filters,
                    '-threads', '2', '-c:v', 'libx264', '-crf', '19', '-movflags', '+faststart',
                    '-t', str(sum(duration for _, duration in entries)), output.name], cwd=directory, check=True)
    encoded_seconds = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', output.name], cwd=directory, text=True).strip())
    manifest = {'encodedDurationSeconds': encoded_seconds, 'video': str(output), 'sourceReceipt': str(directory / 'report.json'),
                'kind': 'sampled actual Electron captures; no anatomy interpolation',
                'nativeTimeStartSeconds': times[0], 'nativeTimeEndSeconds': times[-1],
                'presentationSpeed': speed, 'sampledStates': len(frames),
                'durationSeconds': sum(duration for _, duration in entries),
                'frames': [{k: frame[k] for k in ['id', 'file', 'readoutState', 'readout', 'target', 'fluid', 'position', 'frame']} for frame in frames]}
    (directory / 'recording-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--speed', type=float, default=4.)
    args = parser.parse_args()
    encode(args.directory, args.speed)
