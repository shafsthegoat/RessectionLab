"""Decode selected delivered movie frames, compare their field pixels to desktop captures, and make a review sheet."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(path, number, width, height):
    raw = subprocess.check_output(
        [
            'ffmpeg',
            '-v',
            'error',
            '-nostdin',
            '-threads',
            '1',
            '-ss',
            f'{number/30:.9f}',
            '-i',
            str(path),
            '-frames:v',
            '1',
            '-threads',
            '1',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'rgb24',
            'pipe:1',
        ]
    )
    return Image.frombytes('RGB', (width, height), raw)


def inspect(directory, output, positions, *, partial=False, witness=None):
    package = directory / 'operative-walkthrough.mp4'
    captured = json.loads((directory / 'capture.json').read_text())
    movie = json.loads((directory / 'movie.json').read_text())
    info = json.loads(
        subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_streams', '-show_chapters', '-of', 'json', str(package)]
        )
    )
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    width, height = stream['width'], stream['height']
    if (width, height, stream['codec_name'], stream['avg_frame_rate']) != (1920, 1080, 'h264', '30/1') or (
        not partial and not 60 <= float(stream['duration']) <= 120
    ):
        raise ValueError('Unexpected delivered movie format or duration')
    if (
        digest(package) != movie['videoSha256']
        or movie['captureFileSha256'] != digest(directory / 'capture.json')
        or captured['windowEverVisible']
        or not captured['passed']
    ):
        raise ValueError('Movie, capture hash or hidden-window provenance differs')
    actions = captured['actions']
    if not partial and (
        not captured['completeEpisode']
        or not movie['completeEpisode']
        or len(actions) != captured['steps']
        or len(actions) != movie['actions']
    ):
        raise ValueError('Delivered film does not cover the full saved episode')
    spans = movie['spans']
    count = 0
    for span in spans:
        if span['firstVideoFrame'] != count or span['frames'] <= 0:
            raise ValueError('Overlapping or missing movie frames')
        count += span['frames']
    if count != int(stream['nb_frames']):
        raise ValueError('Decoded stream frame count differs from the retained edit')
    if not positions:
        positions = sorted(
            {chapter['last'] + 1 for chapter in captured['chapters']}
            | {index + 1 for index in captured['plan'].get('stills', [])}
        )
    if not positions or any(n < 1 or n > len(actions) for n in positions):
        raise ValueError('Choose a committed action number in this capture')
    first = {span['captureFrame']: span['firstVideoFrame'] for span in spans}
    if any(row['lastCaptured'] not in first for row in actions):
        raise ValueError('A committed action lost its boundary from the encoded movie')
    selected = [('Overview', next(i for i, x in enumerate(captured['frames']) if x.get('overview')))]
    for position in positions:
        selected.append((f'Action {position}', actions[position - 1]['lastCaptured']))
    rows = []
    thumbs = []
    with (directory / 'frames.mjpg').open('rb') as spool:
        for label, index in selected:
            frame = captured['frames'][index]
            number = first[index]
            decoded = decode(package, number, width, height)
            spool.seek(frame['offset'])
            photo = Image.open(io.BytesIO(spool.read(frame['bytes']))).convert('RGB')
            # The caption bands occupy y=0..111 and y=996..1079 in this format.
            scale = min(width / photo.width, 884 / photo.height)
            size = (round(photo.width * scale), round(photo.height * scale))
            x = (width - size[0]) // 2
            y = 112 + (884 - size[1]) // 2
            actual = decoded.crop((x, y, x + size[0], y + size[1]))
            expected = photo.resize(size, Image.Resampling.LANCZOS)
            # Downsample a few pixels to suppress JPEG/H.264 chroma and ringing differences.
            sample = (max(1, size[0] // 4), max(1, size[1] // 4))
            target = np.asarray(expected.resize(sample, Image.Resampling.BOX), dtype=float)
            measured = np.asarray(actual.resize(sample, Image.Resampling.BOX), dtype=float)
            difference = abs(target - measured)
            occupied = target.max(axis=2) > 35
            mae = float(difference[occupied].mean()) if occupied.any() else float(difference.mean())
            p95 = float(np.percentile(difference, 95))
            if mae > 7 or p95 > 12:
                raise ValueError(
                    f'Encoded field materially differs from capture at {label}: MAE {mae:.3f}, p95 {p95:.3f}'
                )
            rows.append(
                {
                    'sample': label,
                    'captureFrame': index,
                    'videoFrame': number,
                    'seconds': number / 30,
                    'stateId': frame['stateId'],
                    'captureRegionPx': [x, y, *size],
                    'visibleRGBMeanAbsoluteError': mae,
                    'rgbP95AbsoluteError': p95,
                }
            )
            thumbs.append(decoded.resize((480, 270), Image.Resampling.LANCZOS))
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (1440, 302 * ((len(thumbs) + 2) // 3)), '#142225')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    for i, (row, thumb) in enumerate(zip(rows, thumbs, strict=True)):
        x, y = 480 * (i % 3), 302 * (i // 3)
        sheet.paste(thumb, (x, y))
        draw.text((x + 8, y + 275), f"{row['sample']}  ·  {row['seconds']:.2f} s", font=font, fill='#edf4f2')
    sheet.save(output / 'decoded-review.jpg', quality=89)
    report = {
        'videoSHA256': digest(package),
        'captureSHA256': digest(directory / 'capture.json'),
        'sourcePackageSHA256': movie['sourcePackageSha256'],
        'actualStream': {
            k: stream[k] for k in ('width', 'height', 'codec_name', 'avg_frame_rate', 'nb_frames', 'duration')
        },
        'chapters': [c['tags']['title'] for c in info['chapters']],
        'samples': rows,
    }
    if witness:
        from profile_glioma_episode import CaptureAudit

        evidence = json.loads(gzip.decompress(witness.read_bytes()))
        audit = CaptureAudit(directory / 'capture.json', **evidence['identity'])
        if len(evidence['boundaries']) != len(actions) + 1:
            raise ValueError('Cold witness does not cover every recorded action')
        for i, state in enumerate(evidence['boundaries']):
            audit.boundary(state, i)
        report['coldBoundaryWitness'] = {'fileSHA256': digest(witness), **audit.report}
    (output / 'decoded-review.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument(
        '--action',
        dest='actions',
        type=int,
        action='append',
        help='1-based accepted action boundary; repeat as needed.',
    )
    p.add_argument('--partial', action='store_true', help='Permit a shorter early working cut.')
    p.add_argument(
        '--witness', type=Path, help='Require agreement with each independently cold-replayed physical boundary.'
    )
    args = p.parse_args()
    inspect(args.capture, args.output, args.actions, partial=args.partial, witness=args.witness)
