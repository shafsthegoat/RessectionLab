"""Encode a source-bound desktop capture as a 1920×1080, 30 fps rehearsal."""
from __future__ import annotations

from collections import defaultdict

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1920, 1080, 30
FIELD_TOP, FIELD_BOTTOM = 112, 996


def fonts():
    for regular, bold in (
        ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    ):
        if Path(regular).is_file() and Path(bold).is_file():
            return {name: ImageFont.truetype(bold if name == 'heading' else regular, size)
                    for name, size in [('heading', 39), ('label', 23), ('tool', 28)]}
    raise RuntimeError('Install an Arial or DejaVu Sans regular/bold pair for the caption bands.')


def frame_schedule(report):
    plan, frames = report['plan'], report['frames']
    seconds = float(plan.get('presentationSeconds', 90))
    if not 4 <= seconds <= 120:
        raise ValueError('Choose a four- to 120-second presentation (60–120 seconds for the final film).')
    total = round(seconds * FPS)
    intro = min(90, total // 7)
    outro = min(45, total // 10)
    overview = [i for i, f in enumerate(frames) if f.get('overview')]
    transition = [i for i, f in enumerate(frames) if f.get('opticalTransition')]
    if not overview:
        raise ValueError('A real source overview is required.')
    schedule = [overview[0]] * (intro - (intro // 2 if transition else 0))
    if transition:
        schedule.extend(transition[min(len(transition)-1, j*len(transition)//(intro//2))] for j in range(intro//2))

    by_position = defaultdict(list)
    for index, frame in enumerate(frames):
        if 'position' in frame:
            by_position[frame['position']].append(index)
    actions = []
    phase_totals = defaultdict(float)
    for row in report['actions']:
        pos = row['position']
        selected = by_position[pos]
        assert selected and [frames[i]['frameIndex'] for i in selected] == list(range(len(selected)))
        assert frames[selected[-1]]['stateId'] == row['stateId'] == row['native']['stateAfter']
        assert frames[selected[0]]['stateId'] == row['native']['stateBefore']
        duration = max(0., frames[selected[-1]]['time'] - frames[selected[0]]['time'])
        chapter = plan['chapters'][frames[selected[-1]]['chapter']]
        hold = float(plan.get('holdSecondsAtAction', {}).get(str(pos + 1), 0.))
        if not 0 <= hold <= 3:
            raise ValueError('A presentation hold must be between zero and three seconds.')
        cost = chapter.get('weight', 1.) * (duration + .08)
        phase = chapter.get('phase')
        actions.append((selected, duration, cost, round(hold * FPS), phase))
        phase_totals[phase] += cost
    available = total - len(schedule) - outro - sum(a[3] for a in actions)
    # Every action retains a presentation boundary. Faster intervals may sample
    # fewer native microsteps; the origin, endpoint, and full archive stay exact.
    phase_weights = plan.get('phaseWeights')
    if phase_weights:
        actions = [(ids, duration, phase_weights[phase] * cost / phase_totals[phase], hold, phase)
                   for ids, duration, cost, hold, phase in actions]
    minimum = min(1 if phase_weights else 3, available // len(actions))
    if minimum < 1:
        raise ValueError('Presentation is too short to include every committed action.')
    weighted = available - minimum * len(actions)
    weight = sum(a[2] for a in actions)
    cumulative = assigned = 0
    for indices, duration, cost, hold, phase in actions:
        cumulative += cost
        amount = minimum + round(weighted * cumulative / weight) - assigned
        assigned += amount - minimum
        begin = frames[indices[0]]['time']
        step = 0
        for j in range(amount):
            t = begin + duration * j / max(1, amount-1)
            while step+1 < len(indices) and frames[indices[step+1]]['time'] <= t+1e-9:
                step += 1
            schedule.append(indices[-1] if j+1 == amount else indices[step])
        schedule.extend([indices[-1]] * hold)
    schedule += [actions[-1][0][-1]] * outro
    assert len(schedule) == total
    assert all(a <= b for a, b in zip(schedule,schedule[1:]))
    return schedule


def render(photo, frame, plan, font, instruments, movie_second):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), '#0d191d')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, FIELD_TOP, WIDTH, FIELD_BOTTOM), fill='#030708')
    drawing = photo.convert('RGB')
    factor = min(WIDTH/drawing.width, (FIELD_BOTTOM-FIELD_TOP)/drawing.height)
    drawing = drawing.resize((round(drawing.width*factor),round(drawing.height*factor)), Image.Resampling.LANCZOS)
    canvas.paste(drawing, ((WIDTH-drawing.width)//2, FIELD_TOP+(FIELD_BOTTOM-FIELD_TOP-drawing.height)//2))
    overview = frame.get('overview') or frame.get('opticalTransition')
    name = (plan.get('overviewTitle','Anatomical overview') if frame.get('overview') else 'Operative approach') if overview else plan['chapters'][frame['chapter']]['title']
    draw.text((38,14),plan['anatomyLabel'],fill='#a7ccbd',font=font['label'])
    draw.text((36,48),name,fill='#f4f4e9',font=font['heading'])
    t=frame['time']
    draw.text((1882,20),f'Offline replay  ·  Movie {movie_second//60:02}:{movie_second%60:02}',anchor='ra',fill='#c1d0ce',font=font['label'])
    draw.text((1882,60),f'Simulation   {int(t)//60:02}:{t%60:04.1f}',anchor='ra',fill='#d9e7e2',font=font['tool'])
    if overview:
        draw.text((38,1022),plan.get('overviewLabel','Registered anatomy before intervention'),fill='#e0e7dd',font=font['tool'])
    else:
        for x,hand in ((38,'left'),(1000,'right')):
            tool=frame['tools'][hand]
            display=instruments.get(tool['instrument']) or tool['instrument'].replace('_',' ').capitalize()
            state='withdrawn' if not tool['inserted'] else ('active' if frame.get('active',{}).get(hand) else 'positioned')
            for name, body in (frame.get('exposure') or {}).get('bodies', {}).items():
                if body['heldBy'] == hand:
                    state = 'holding ' + name
                    break
            draw.text((x,1007),hand.title(),fill='#a7ccbd',font=font['label'])
            draw.text((x+79,1003),f'{display}  ·  {state}',fill='#eef1e7',font=font['tool'])
        if plan.get('actions'):
            description=plan['actions'][frame['position']]['text']
            draw.text((38,1044),description,fill='#adbfba',font=font['label'])
    stream=io.BytesIO();canvas.save(stream,format='JPEG',quality=94,subsampling=0)
    return stream.getvalue()


def encode(directory):
    report=json.loads((directory/'capture.json').read_text())
    assert report['passed'] and not report['windowEverVisible']
    if hashlib.sha256(Path(report['caseFile']).read_bytes()).hexdigest()!=report['sourceFileSha256']:
        raise ValueError('Saved source no longer matches the captured episode.')
    schedule=frame_schedule(report)
    target=directory/'operative-walkthrough.mp4'
    if target.exists():
        raise FileExistsError(target)
    font=fonts(); spans=[]
    for i,selected in enumerate(schedule):
        if not spans or spans[-1]['captureFrame']!=selected or i % FPS == 0:
            spans.append({'captureFrame':selected,'firstVideoFrame':i,'frames':0})
        spans[-1]['frames']+=1
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        take=Path(scratch)/'encode.mp4'
        chapter_file=Path(scratch)/'chapters.txt'
        entries=[]
        for n,frame_index in enumerate(schedule):
            row=report['frames'][frame_index]
            title=report['plan'].get('overviewTitle','Anatomical overview') if row.get('overview') else 'Operative approach' if row.get('opticalTransition') else report['plan']['chapters'][row['chapter']]['title']
            if not entries or entries[-1][1]!=title:entries.append((n,title))
        chapters=[';FFMETADATA1']
        for i,(start,title) in enumerate(entries):
            end=entries[i+1][0] if i+1<len(entries) else len(schedule)
            escaped=title.replace('\\','\\\\').replace('=','\\=').replace(';','\\;').replace('#','\\#').replace('\n',' ')
            chapters.extend(['[CHAPTER]',f'TIMEBASE=1/{FPS}',f'START={start}',f'END={end}','title='+escaped])
        chapter_file.write_text('\n'.join(chapters)+'\n')
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','image2pipe','-vcodec','mjpeg','-framerate',str(FPS),
                 '-i','pipe:0','-f','ffmetadata','-i',str(chapter_file),'-map','0:v','-map_chapters','1','-an','-c:v','libx264',
                 '-preset','veryfast','-crf','18','-threads','3','-pix_fmt','yuv420p','-movflags','+faststart',str(take)]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            with (directory/'frames.mjpg').open('rb') as source:
                for span in spans:
                    row=report['frames'][span['captureFrame']]
                    source.seek(row['offset']);data=source.read(row['bytes']);assert len(data)==row['bytes']
                    with Image.open(io.BytesIO(data)) as scene:frame=render(scene,row,report['plan'],font,report.get('instruments', {}),span['firstVideoFrame']//FPS)
                    for _ in range(span['frames']):proc.stdin.write(frame)
            proc.stdin.close()
            if proc.wait():raise RuntimeError('FFmpeg failed to encode the checked observations.')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
        checked=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
          'stream=width,height,codec_name,r_frame_rate,nb_frames,duration','-of','json',str(take)]))['streams'][0]
        assert (checked['width'],checked['height'],checked['codec_name'],checked['r_frame_rate'])==(WIDTH,HEIGHT,'h264','30/1')
        assert int(checked['nb_frames'])==len(schedule)
        take.rename(target)
    result={'sourcePackageSha256':report['sourceFileSha256'],'branchId':report['branchId'],'captureFileSha256':hashlib.sha256((directory/'capture.json').read_bytes()).hexdigest(),
            'completeEpisode':report['completeEpisode'],'simulatedSeconds':report['actions'][-1]['time'],'stream':checked,
            'nativeObservations':sum(not f.get('overview') and not f.get('opticalTransition') for f in report['frames']),
            'actions':len(report['actions']),'method':'Offline presentation of GPU-completed native desktop observations. No geometry or tool-pose interpolation. Time labels are native simulation time. Repeated frames preserve a displayed state.',
            'videoSha256':hashlib.sha256(target.read_bytes()).hexdigest(),'spans':spans}
    (directory/'movie.json').write_text(json.dumps(result,indent=2)+'\n')
    print(target,checked['duration'],result['actions'],result['nativeObservations'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    encode(parser.parse_args().directory)
