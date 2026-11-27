"""Compare recorded phases across cases, then matched native states in the same field."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from encode_surgical_walkthrough import FIELD_BOTTOM, FIELD_TOP, FPS, HEIGHT, WIDTH, fonts


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_comparison(capture, baseline, plan_path, destination):
    plan = json.loads(plan_path.read_text())
    report = json.loads((capture / 'capture.json').read_text())
    movie = json.loads((capture / 'movie.json').read_text())
    new_video = capture / 'operative-walkthrough.mp4'
    if sha(baseline) != plan['baselineSha256'] or sha(new_video) != movie['videoSha256']:
        raise ValueError('Input movie differs from the reviewed recording.')
    if destination.exists():
        raise FileExistsError(destination)
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_chapters', '-of', 'json', str(baseline)]))
    chapters = {c['tags']['title']: c for c in metadata['chapters']}
    font = fonts()
    panel_width, panel_height = 932, 690
    seconds = 3
    duration = seconds * FPS
    segments = []

    def draw_labels(title, left, right, footer, file):
        canvas = Image.new('RGB', (WIDTH, HEIGHT), '#101c20')
        draw = ImageDraw.Draw(canvas)
        draw.text((34, 23), 'RessectionLab   /   Recorded rehearsal comparison', fill='#adcebd', font=font['label'])
        draw.text((32, 61), title, fill='#f7f5e9', font=font['heading'])
        draw.text((34, 145), left, fill='#f2f1e8', font=font['tool'])
        draw.text((978, 145), right, fill='#f2f1e8', font=font['tool'])
        draw.text((34, 993), footer, fill='#bfd4c9', font=font['label'])
        canvas.save(file)

    with tempfile.TemporaryDirectory(dir=destination.parent) as name:
        scratch = Path(name)
        with (capture / 'frames.mjpg').open('rb') as spool:
            for i, shot in enumerate(plan['shots']):
                labels = scratch / f'labels-{i}.png'
                start_flags = ['-loop', '1', '-framerate', str(FPS)]
                input_command = ['-loop', '1', '-framerate', str(FPS), '-i', str(labels)]
                after_crop = ''
                if 'matchedBiteDirectory' in shot:
                    folder = Path(shot['matchedBiteDirectory'])
                    builds = json.loads((folder / 'builds.json').read_text())
                    matched = json.loads((folder / 'comparison.json').read_text())['bite-1440']
                    if not matched['cameraAndStateMatched'] or matched['beforeViewport'] != matched['afterViewport']:
                        raise ValueError('The BITE pair is not camera and state matched.')
                    checks = {line.split(maxsplit=1)[1].lstrip('*'): line.split(maxsplit=1)[0]
                              for line in (folder / 'SHA256SUMS').read_text().splitlines() if line.strip()}
                    originals = []
                    for side in ('before', 'after'):
                        relative = f"{side}-bite/{matched['frame']}.png"
                        source = folder / relative
                        digest = sha(source)
                        if digest != checks.get(relative):
                            raise ValueError('Matched BITE source frame checksum differs.')
                        picture = scratch / f'{side}-bite.png'
                        with Image.open(source) as image:
                            image.crop(matched['roiScreenPx']).save(picture)
                        input_command += start_flags + ['-i', str(picture)]
                        originals.append({'path': str(source), 'sha256': digest, 'renderer': builds[side]})
                    if builds['bite']['snapshot'] != matched['source']['snapshot']:
                        raise ValueError('The matched BITE images refer to a different source.')
                    before_revision = builds['before']['baselineCommit'][:8]
                    changed = ' + '.join(builds['after']['geometryCommits'])
                    draw_labels(shot['title'], 'Earlier renderer  /  ' + before_revision,
                                'Revised geometry  /  ' + changed,
                                'Historical BITE-09 render comparison. Identical source, completed dural opening, camera and scale.', labels)
                    evidence = {'sameSource': builds['bite'], 'frames': originals, 'matched': matched}
                elif 'matchedCaptureBoundary' in shot:
                    paired = report['matchedFrames'][str(shot['matchedCaptureBoundary'])]
                    for side in ('before', 'after'):
                        picture = Path(paired[side+'File'])
                        if sha(picture) != paired[side+'Sha256']:
                            raise ValueError('The native comparison images changed after capture.')
                        input_command += start_flags + ['-i', str(picture)]
                    draw_labels(shot['title'], 'Earlier operative renderer', 'Current operative renderer',
                                'Same generated case and accepted action. Camera, optical field, tools and tissue state match.', labels)
                    evidence = {'matchedNativeRevisions': paired}
                elif 'baselineChapter' in shot:
                    chapter = chapters[shot['baselineChapter']]
                    begin, end = float(chapter['start_time']), float(chapter['end_time'])
                    old_start = max(begin, (begin + end - seconds) / 2)
                    # Derive the appearance of this accepted action in the final movie from its exact capture-frame index.
                    endpoint = report['actions'][shot['action'] - 1]
                    selected = next(s for s in movie['spans'] if s['captureFrame'] == endpoint['lastCaptured'])
                    new_start = max(0., min(float(movie['stream']['duration']) - seconds, selected['firstVideoFrame'] / FPS - seconds / 2))
                    input_command += ['-ss', str(old_start), '-i', str(baseline), '-ss', str(new_start), '-i', str(new_video)]
                    draw_labels(shot['title'], 'BITE-09  /  Four-tool geometric preview',
                                'Generated glioma  /  Accepted instrument actions',
                                'Phase comparison across different cases. Anatomy, optical scale and timing differ.', labels)
                    after_crop = f'crop=iw:{FIELD_BOTTOM-FIELD_TOP}:0:{FIELD_TOP},'
                    evidence = {'baselineChapter': shot['baselineChapter'], 'baselineStartSeconds': old_start,
                                'generatedStartSeconds': new_start, 'stateAnchor': endpoint['stateId'], 'action': shot['action']}
                else:
                    rows = []
                    for side, pos in enumerate(shot['matchedActions']):
                        row = report['frames'][report['actions'][pos - 1]['lastCaptured']]
                        rows.append(row)
                        spool.seek(row['offset'])
                        picture = scratch / f'state-{i}-{side}.jpg'
                        picture.write_bytes(spool.read(row['bytes']))
                        input_command += start_flags + ['-i', str(picture)]
                    for key in ('fieldMm', 'width', 'height'):
                        if rows[0][key] != rows[1][key]:
                            raise ValueError(f'Same-field shots have different {key} at {shot["matchedActions"]}.')
                    max_delta=max(abs(x-y) for key in ('target','position','up')
                                  for x,y in zip(rows[0]['camera'][key],rows[1]['camera'][key],strict=True))
                    if max_delta>1e-6:
                        raise ValueError('The same-source operative microscope moved between matched states.')
                    draw_labels(shot['title'], 'Before  /  ' + shot['left'], 'After  /  ' + shot['right'],
                                'Same generated source, branch, camera and optical field. Both images are accepted operative views.', labels)
                    evidence = {'sameSource': report['sourceCaseHash'], 'camera': rows[0]['camera'],
                                'fieldMm': rows[0]['fieldMm'], 'maxCameraComponentDelta': max_delta, 'states': [r['stateId'] for r in rows]}
                dims = f'scale={panel_width}:{panel_height}:force_original_aspect_ratio=decrease,pad={panel_width}:{panel_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1'
                filters = f'[1:v]{dims}[a];[2:v]{after_crop}{dims}[b];[0:v][a]overlay=20:215[left];[left][b]overlay=968:215[out]'
                segment = scratch / f'comparison-{i:02}.mp4'
                subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', *input_command,
                                '-filter_complex_threads', '2', '-filter_complex', filters, '-map', '[out]', '-an',
                                '-frames:v', str(duration), '-r', str(FPS), '-c:v', 'libx264', '-threads', '3',
                                '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '18', str(segment)], check=True)
                segments.append({'file': segment.name, 'title': shot['title'], **evidence})
        listing = scratch / 'segments.txt'
        listing.write_text(''.join(f"file '{s['file']}'\n" for s in segments))
        target = scratch / 'comparison.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'concat', '-safe', '1',
                        '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(target)], check=True)
        checked = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height,codec_name,nb_frames,r_frame_rate,duration', '-of', 'json', str(target)]))['streams'][0]
        assert (checked['width'], checked['height'], checked['r_frame_rate'], checked['codec_name']) == (WIDTH, HEIGHT, '30/1', 'h264')
        assert int(checked['nb_frames']) == duration * len(segments)
        target.rename(destination)
    result = {'baseline': str(baseline), 'baselineSha256': plan['baselineSha256'], 'generatedMovieSha256': movie['videoSha256'],
              'comparisonSha256': sha(destination), 'stream': checked, 'segments': segments}
    destination.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(destination, checked['duration'], 'seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'baseline', 'plan', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run_comparison(args.capture, args.baseline, args.plan, args.output)
