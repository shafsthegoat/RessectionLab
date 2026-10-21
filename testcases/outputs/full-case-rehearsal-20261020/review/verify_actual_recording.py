#!/usr/bin/env python3
"""Offline CUA recording/log/AX checks and optional lossless MP4 remux.
No app, screenshot, native engine, replay or synthetic video is created.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import datetime

FFMPEG = '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/opt/homebrew/bin/ffprobe'
SOURCE = 'CUA App.getScreenshot({emit:false}) only'
RANGES = {'primary': (0, 2148), 'vascular': (42, 73)}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def ignored(path):
    path = path.resolve()
    require('/testcases/outputs/' in str(path), 'Output must be under ignored testcases/outputs/')
    return path

def actual_only(*values):
    require(all('ENGINEERINGHELPERTEST' not in str(value).upper() for value in values),
            'ENGINEERINGHELPERTEST is never actual recording evidence or handoff input')

def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()

def checked_log(rows, recording):
    actual_only(recording)
    require(rows and rows[0]['kind'] == 'start' and rows[-1]['kind'] == 'stop', 'Require complete start/stop capture log')
    starts = [row for row in rows if row['kind'] == 'start']
    encoders = [row for row in rows if row['kind'] == 'encoder_start']
    require(len(starts) == len(encoders) == 1, 'Require one recorder/encoder session')
    require(starts[0].get('engineeringTest') is False and encoders[0].get('captureSource') == SOURCE, 'Actual CUA source declaration required')
    require(Path(starts[0]['outputPath']).resolve() == recording.resolve(), 'Capture log belongs to a different video path')
    require(not any('error' in row['kind'] for row in rows), 'Capture/encoder error event in log')
    stop = rows[-1]
    state = stop['state']
    require(state.get('engineeringTest') is False and state['captureErrors'] == state['encoderErrors'] == 0,
            'Engineering/error state cannot be handed off')
    require(state['stopped'] is True and stop['result']['code'] == state['ffmpegExitCode'] == 0, 'Encoder did not finalize successfully')
    require(state.get('lastError') is None, 'Recorder reports an error')
    require(all(rows[i]['monotonicMs'] <= rows[i + 1]['monotonicMs'] for i in range(len(rows) - 1)), 'Nonmonotonic log clock')
    captures = [row for row in rows if row['kind'] == 'capture']
    streamed = [row for row in rows if row['kind'] == 'frame_streamed']
    require(len(captures) >= 2, 'Need at least two actual captured frames')
    expected = list(range(1, len(captures) + 1))
    require([row['captureNumber'] for row in captures] == [row['captureNumber'] for row in streamed] == expected,
            'Missing/reordered captured or streamed frame')
    require(state['captures'] == state['encodedInputFrames'] == len(captures), 'Final capture/stream counts disagree')
    require(captures[-1].get('final') is True and sum(bool(row.get('final')) for row in captures) == 1, 'Require one final actual screenshot')
    for row in captures:
        require(row['begin']['monotonicMs'] <= row['end']['monotonicMs'], 'Capture end precedes its start')
        require(row['codec'] in ('png', 'mjpeg') and row['bytes'] > 0 and row['width'] > 0 and row['height'] > 0, 'Invalid screenshot metadata')
    intervals = []
    opened = None
    for row in rows:
        if row['kind'] == 'ui_begin':
            require(opened is None, 'Overlapping UI operations')
            opened = row['monotonicMs']
        elif row['kind'] == 'ui_end':
            require(opened is not None, 'Unmatched UI end')
            intervals.append((opened, row['monotonicMs']))
            opened = None
    require(opened is None, 'UI interval left open')
    for row in captures:
        begin, end = row['begin']['monotonicMs'], row['end']['monotonicMs']
        require(not any(max(begin, left) < min(end, right) for left, right in intervals), 'Screenshot overlaps serialized UI operation')
    for left, right in zip(captures, captures[1:]):
        require(left['end']['monotonicMs'] <= right['begin']['monotonicMs'], 'Overlapping screenshot calls')
    flags = encoders[0]['args']
    require('-use_wallclock_as_timestamps' in flags and flags[flags.index('-use_wallclock_as_timestamps') + 1] == '1', 'Missing wall-clock timestamp input')
    require('-fps_mode' in flags and flags[flags.index('-fps_mode') + 1] == 'vfr' and '-r' not in flags, 'Unexpected fixed-rate/drop/dup encoding policy')
    return captures, dict(capturedFrames=len(captures), uiIntervals=len(intervals),
        captureSpanSeconds=(captures[-1]['end']['monotonicMs'] - captures[0]['end']['monotonicMs']) / 1000,
        maxCaptureCallSeconds=max((row['end']['monotonicMs'] - row['begin']['monotonicMs']) / 1000 for row in captures),
        maxBackpressureSeconds=max(row['backpressureMs'] for row in streamed) / 1000)

def probe(path):
    result = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_packets', '-show_streams',
        '-show_entries', 'packet=pts_time,dts_time,duration_time:stream=codec_name,profile,level,width,height,time_base', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    require(not result.stderr.strip(), 'ffprobe reported container/probe errors: ' + result.stderr[:1000])
    value = json.loads(result.stdout)
    require(len(value.get('streams', [])) == 1 and value['streams'][0]['codec_name'] == 'h264', 'Require one selected H264 video stream')
    pts = [float(packet['pts_time']) for packet in value.get('packets', [])]
    require(len(pts) >= 2 and all(math.isfinite(value) for value in pts), 'Missing/nonfinite packet PTS')
    require(all(right > left for left, right in zip(pts, pts[1:])), 'Packet PTS are not strictly increasing')
    return pts, value['streams'][0]

def checked_observations(document, recording, captures, segment):
    actual_only(json.dumps(document))
    require(document['schema'] == 'resectionlab.actual-video-observations.v1', 'Unknown observation receipt schema')
    require(Path(document['recordingPath']).resolve() == recording.resolve(), 'Observations bind a different recording')
    require(document['segment'] == segment and (document['fromAction'], document['toAction']) == RANGES[segment], 'Wrong requested segment range')
    require(isinstance(document['branchText'], str) and len(document['branchText']) > 0, 'Require actual nonempty branch text')
    require(document.get('continuousPlaybackObserved') is True and document.get('seekOrBranchChangesDuringSegment') is False,
            'Root must explicitly attest continuous playback without seeks/branch changes inside the requested segment')
    native = Path(document['nativePath']).resolve()
    require(digest(native) == document['nativeSha256'], 'Observed frozen native hash differs')
    observations = document['observations']
    tolerance = document.get('captureBindingToleranceSeconds', 5)
    require(isinstance(tolerance, (int, float)) and 0 < tolerance <= 5, 'AX/capture binding tolerance must be <=5 seconds')
    require(len(observations) >= 3, 'Require actual start, progress, and endpoint observations')
    bounds, moments, capture_numbers, refs = [], [], [], []
    for row in observations:
        ax = Path(row['axPath']).resolve()
        screenshot = Path(row['screenshotPath']).resolve()
        require(ax.is_file() and screenshot.is_file(), 'Missing actual AX/screenshot observation')
        text = ax.read_text()
        match = re.search(r'Recorded action boundary\s+(\d+)\s*/\s*(\d+)', text)
        require(match and int(match.group(1)) == row['actionBoundary'], 'AX boundary does not match claimed observation')
        require(int(match.group(2)) == RANGES[segment][1], 'AX action total differs from requested segment endpoint')
        require(document['branchText'] in text, 'Observed AX does not show the declared branch text')
        number = row['captureNumber']
        require(isinstance(number, int) and 1 <= number <= len(captures), 'Observation requires an actual recorder capture number')
        moment = utc(row['utc'])
        require(abs(moment - utc(captures[number - 1]['end']['utc'])) <= tolerance,
                'AX observation time is not close to the claimed actual capture')
        bounds.append(row['actionBoundary']);moments.append(moment);capture_numbers.append(number)
        refs.append(dict(axPath=str(ax), axSha256=digest(ax), screenshotPath=str(screenshot), screenshotSha256=digest(screenshot),
                         captureNumber=number, actionBoundary=row['actionBoundary'], utc=row['utc']))
    require(bounds[0] == RANGES[segment][0] and bounds[-1] == RANGES[segment][1], 'Missing requested start/end AX boundary')
    require(any(RANGES[segment][0] < boundary < RANGES[segment][1] for boundary in bounds[1:-1]), 'Require a strictly interior observed action boundary')
    require(all(a <= b for a, b in zip(bounds, bounds[1:])), 'Observation boundaries run backwards')
    require(all(a < b for a, b in zip(moments, moments[1:])), 'AX anchor UTC times must strictly increase')
    require(all(a < b for a, b in zip(capture_numbers, capture_numbers[1:])), 'AX anchor capture numbers must strictly increase')
    return dict(segment=segment, fromAction=bounds[0], toAction=bounds[-1], nativePath=str(native), nativeSha256=document['nativeSha256'],
        observations=refs, continuousPlaybackAttestation=True,
        qualification='AX anchors and root continuous-playback attestation; codec/frame checks alone do not prove every timeline state appeared.')

def faststart(path):
    boxes = []
    with path.open('rb') as stream:
        total = path.stat().st_size
        while stream.tell() + 8 <= total:
            start = stream.tell();header = stream.read(8);size = int.from_bytes(header[:4], 'big');kind = header[4:].decode('ascii', 'replace')
            if size == 1:
                size = int.from_bytes(stream.read(8), 'big')
            if size == 0:
                size = total - start
            require(size >= 8 and start + size <= total, 'Invalid MP4 box')
            boxes.append(kind);stream.seek(start + size)
    require('moov' in boxes and 'mdat' in boxes and boxes.index('moov') < boxes.index('mdat'), 'MP4 is not faststart')
    return boxes

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--segment', choices=RANGES, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--mp4', type=Path)
    parser.add_argument('--timing-tolerance-seconds', type=float, default=2)
    args = parser.parse_args()
    recording = args.recording.resolve();log = (args.log or Path(str(recording) + '.capture.jsonl')).resolve()
    actual_only(recording, log, args.receipt, args.mp4 or '')
    receipt = ignored(args.receipt);require(not receipt.exists(), 'Receipt already exists')
    require(0 <= args.timing_tolerance_seconds <= 5, 'Timing tolerance must be within0–5 seconds')
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns, digest(path)) for path in (recording, log)}
    def stable_inputs():
        for path, identity in before.items():
            require((path.stat().st_size, path.stat().st_mtime_ns, digest(path)) == identity, 'Input changed during offline verification')
    captures, counts = checked_log([json.loads(line) for line in log.read_text().splitlines() if line], recording)
    pts, stream = probe(recording)
    require(len(pts) == counts['capturedFrames'], 'Encoded packet count differs from actual capture/stream count')
    span = pts[-1] - pts[0]
    require(abs(pts[0]) <= .01, 'Unexpected nonzero recording PTS origin')
    require(abs(span - counts['captureSpanSeconds']) <= args.timing_tolerance_seconds, 'Video PTS span differs from actual capture span beyond allowed tolerance')
    timeline = None
    if args.observations:
        timeline = checked_observations(json.loads(args.observations.read_text()), recording, captures, args.segment)
    result = dict(schema='resectionlab.actual-recording-verification.v1', status='media-and-AX-anchors-verified' if timeline else 'media-only-timeline-unverified',
        recordingPath=str(recording), recordingSha256=before[recording][2], captureLogPath=str(log), captureLogSha256=before[log][2],
        stream=stream, **counts, packetCount=len(pts), packetPtsSpanSeconds=span, timingToleranceSeconds=args.timing_tolerance_seconds,
        timeline=timeline, qualification='Actual CUA log declarations + media packet checks; no engine replay or full pixel decode performed. Demux-read PTS can differ from exact screenshot time.')
    if args.mp4:
        require(timeline is not None, 'Final MP4 requires explicit actual AX observation receipt')
        output = ignored(args.mp4);require(output.suffix.lower() == '.mp4' and not output.exists(), 'MP4 path must be new/exclusive')
        # Private same-filesystem directory + atomic exclusive hard-link install:
        # ffmpeg faststart needs a seekable path; no input/output is overwritten.
        with tempfile.TemporaryDirectory(prefix='actual-remux-', dir=output.parent) as temporary:
            temp = Path(temporary) / 'complete.mp4'
            command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(recording), '-map', '0:v:0',
                       '-c:v', 'copy', '-an', '-copyts', '-start_at_zero', '-movflags', '+faststart', str(temp)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            out_pts, out_stream = probe(temp)
            require(len(out_pts) == len(pts) and all(abs(a - b) <= .0011 for a, b in zip(pts, out_pts)), 'Remux changed packet count/timestamps')
            boxes = faststart(temp)
            stable_inputs()
            os.link(temp, output)
            result['mp4'] = dict(path=str(output), sha256=digest(output), codecCopy=True, packetCount=len(out_pts), stream=out_stream, boxes=boxes, command=command)
    stable_inputs()
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2);stream.write('\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
