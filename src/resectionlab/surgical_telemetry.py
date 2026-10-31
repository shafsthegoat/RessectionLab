"""Original, descriptive bimanual telemetry from committed simulator microsteps.

Inspired by Yilmaz et al. (2022), doi:10.1038/s41746-022-00596-8.
This module supplies no expertise score, learned model, or clinical inference.
"""
from __future__ import annotations

import bisect
import copy
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from .surgical_engine import SurgicalEngine

SCHEMA = 'resectionlab.surgical-telemetry.v1'
LABELS = ('manual', 'scripted', 'search', 'RL')
REFERENCE = {
    'citation': 'Yilmaz R et al. (2022). Continuous monitoring of surgical bimanual expertise using deep neural networks in virtual reality simulation. npj Digital Medicine 5, 54.',
    'doi': 'https://doi.org/10.1038/s41746-022-00596-8',
    'code': 'https://github.com/recaiyilmaz/icems',
    'sampleData': 'https://doi.org/10.6084/m9.figshare.15132507.v1',
    'adaptation': 'Feature selection and 0.2-second reporting inspiration; original implementation, no transferred classifier or reference thresholds.',
}
PROVENANCE = {
    'time': 'simulated elapsed seconds; successful engine microsteps, not wall clock',
    'pose': 'model-derived committed tool-tip RAS+ millimeters and xyzw unit quaternion',
    'active': 'effective activation during the observed microstep; latchedActive records the persistent switch separately',
    'motion': 'finite differences of committed microstep poses with actual dt; path excludes placement jumps; unavailable derivatives are null',
    'sampling': 'last observation at or before each reporting time, no interpolation; observedTimeSeconds and observationAgeSeconds expose staleness',
    'force': 'forceAssumedN is the declared simulator parameter/default, never a measured contact force; measuredForceN is unknown (null)',
    'removal': 'model-derived whole-cell removal, using explicit target membership and affine voxel volume; non-target includes access bone and other materials, not a healthy-tissue injury label',
    'woundsAndFluid': 'model-derived under declared research assumptions; fluid includes irrigation and modeled sources, not measured blood loss',
    'contact': 'model-derived cumulative cell-seconds; exposureByMaterialCellSeconds follows engine working-contact accounting, not complete per-hand force sensing',
    'events': 'committed microstep events have observed simulation times; rejection/contact-stop and exchange outcomes are action-boundary events',
    'injuryAndExpertise': 'clinical injury, patient risk, and expertise are unknown; no score inferred',
}


def _norm(value):
    return float(np.linalg.norm(value))


class _ObservedEngine(SurgicalEngine):
    """Replay observer with its own transactional state, outside engine identity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._telemetry = []
        self._attempts = []
        self._events = []
        self._path = {hand: 0. for hand in ('left', 'right')}
        self._reposition = {hand: 0. for hand in ('left', 'right')}
        self._previous = {}
        self._invalid = 0
        self._exchanges = 0
        self._contact_stops = 0
        self._telemetry.append(self._sample('initial'))

    def clone(self):
        other = super().clone()
        # Samples are immutable after append; only small derivative state needs
        # a deep copy. Rejected candidate actions must discard every sample.
        for name in ('_telemetry', '_attempts', '_events'):
            setattr(other, name, getattr(self, name)[:])
        for name in ('_path', '_reposition', '_previous'):
            setattr(other, name, copy.deepcopy(getattr(self, name)))
        return other

    def _sample(self, kind, active_hand=None, effective_active=False, before=None, dt=None):
        state = self.state
        slots = {}
        for hand, slot in state['slots'].items():
            position = np.asarray(slot['position'], float)
            velocity = acceleration = speed = None
            discontinuity = False
            if before is not None:
                start = np.asarray(before[hand]['position'], float)
                previous = self._previous.get(hand)
                discontinuity = previous is not None and (
                    _norm(start - previous['position']) > 1e-8 or slot['instrument'] != previous['instrument'])
                if previous is not None:
                    self._reposition[hand] += _norm(start - previous['position'])
                self._path[hand] += _norm(position - start)
                if not discontinuity:
                    velocity = ((position - start) / dt).tolist()
                    speed = _norm(velocity)
                    if previous is not None and previous['velocity'] is not None:
                        acceleration = ((np.asarray(velocity) - previous['velocity']) / ((dt + previous['dt']) / 2)).tolist()
                self._previous[hand] = {'position': position.tolist(), 'instrument': slot['instrument'],
                                        'velocity': velocity, 'dt': dt}
            elif kind == 'initial':
                self._previous[hand] = {'position': position.tolist(), 'instrument': slot['instrument'], 'velocity': None, 'dt': None}
            elif self._telemetry:
                prior = self._telemetry[-1]['slots'][hand]
                velocity, acceleration, speed = prior['velocityMmPerSecond'], prior['accelerationMmPerSecond2'], prior['speedMmPerSecond']
                discontinuity = prior['poseDiscontinuity']
            active = bool(effective_active if hand == active_hand else slot['active'])
            if kind == 'action_end' and self._telemetry:
                active = self._telemetry[-1]['slots'][hand]['active']
            slots[hand] = {
                'instrument': slot['instrument'], 'positionMm': position.tolist(),
                'quaternionXyzw': list(slot['quaternion']), 'active': active,
                'latchedActive': bool(slot['active']), 'inserted': bool(slot['inserted']),
                'pathLengthMm': self._path[hand], 'repositionDistanceMm': self._reposition[hand],
                'velocityMmPerSecond': velocity, 'speedMmPerSecond': speed,
                'accelerationMmPerSecond2': acceleration,
                'accelerationMagnitudeMmPerSecond2': None if acceleration is None else _norm(acceleration),
                'poseDiscontinuity': discontinuity, 'measuredForceN': None,
                'forceAssumedN': self.parameters(slot).get('forceN'),
            }
        target = sum(bool(self.by_label[int(self.scene.labels.ravel()[cell])].get('target')) for cell in state['removed'])
        wound_sources = [source for source in state['sources'] if source.get('kind') == 'modeled-vessel-wound']
        wounds = state.get('vesselsV2', {}).get('wounds', state.get('vesselWounds'))
        if 'vascularGraphV1' in state: wounds = state['vascularGraphV1']['ports']
        from .surgical_bipolar_thermal import metrics as bipolar_thermal_metrics
        return {
            'timeSeconds': float(state['elapsedSeconds']), 'sampleKind': kind,
            'actionIndex': len(self._attempts), 'slots': slots,
            'kinematicIntervalSeconds': dt if dt is not None else (self._telemetry[-1].get('kinematicIntervalSeconds') if self._telemetry else None),
            'tipSeparationMm': _norm(np.asarray(slots['left']['positionMm']) - slots['right']['positionMm']),
            'targetRemovedMm3': target * self.scene.voxel_mm3,
            'nonTargetRemovedMm3': (len(state['removed']) - target) * self.scene.voxel_mm3,
            'woundCount': None if wounds is None else len(wounds),
            'activeWoundCount': None if wounds is None else sum(bool(source['bleedingActive']) for source in wound_sources),
            'modeledWoundEmittedMm3': None if wounds is None else sum(source.get('emittedMm3', 0.) for source in wound_sources),
            'fluidPresentMm3': sum(state['fluid'].values()) + sum(source.get('reservoirMm3', 0.) for source in state['sources']),
            'fluidAddedMm3': state['fluidAddedMm3'], 'fluidRemovedMm3': state['fluidRemovedMm3'],
            'fluidEscapedMm3': state['fluidEscapedMm3'], 'debrisMm3': state['debrisMm3'],
            'contactCellSeconds': sum(state['contact'].values()),
            'exposureByMaterialCellSeconds': dict(state['materialExposure']),
            'vesselEncounterCells': len(state['vesselEncounters']),
            'invalidActionsCumulative': self._invalid, 'toolExchangesCumulative': self._exchanges,
            'contactStopsCumulative': self._contact_stops,
            **({'bipolarThermal': bipolar_thermal_metrics(self)}
               if 'bipolarThermalV1' in state else {}),
        }

    def _substep(self, key, op, p, q, active, dt):
        before = {hand: {'position': slot['position'][:]} for hand, slot in self.state['slots'].items()}
        events = super()._substep(key, op, p, q, active, dt)
        collector = getattr(self, '_frame_collector', None)
        if collector is not None:
            collector.observe(self, key, bool(active))
        self._telemetry.append(self._sample('microstep', key, active, before, dt))
        self._events.extend({'timeSeconds': self.state['elapsedSeconds'], 'actionIndex': len(self._attempts),
                             'timing': 'committed_microstep', 'event': copy.deepcopy(event)} for event in events)
        return events

    def step(self, action):
        start = self.state['elapsedSeconds']
        collector = getattr(self, '_frame_collector', None)
        try:
            if collector is not None:
                collector.begin(self, action)
            result = super().step(action)
            if collector is not None:
                collector.end(self, action, result)
        except BaseException:
            if collector is not None:
                collector.abort()
            raise
        end = self.state['elapsedSeconds']
        self._invalid += result['status'] == 'rejected'
        self._exchanges += result['status'] != 'rejected' and action.get('operation') == 'exchange'
        self._contact_stops += sum(event['kind'] == 'contact_stop' for event in result.get('events', []))
        index = len(self._attempts)
        attempt = {'actionIndex': index, 'startTimeSeconds': start, 'endTimeSeconds': end,
                   'status': result['status'], 'reason': result.get('reason'),
                   'reasonCode': result.get('reasonCode'), 'completedFraction': result.get('completedFraction'),
                   'action': copy.deepcopy(action), 'stateId': result['stateId']}
        self._telemetry.append(self._sample('action_end'))
        self._attempts.append(attempt)
        self._events.append({'timeSeconds': end, 'actionIndex': index, 'timing': 'action_boundary',
                             'event': {'kind': 'action_outcome', 'status': result['status'],
                                       'reason': result.get('reason'), 'operation': action.get('operation'), 'slot': action.get('slot')}})
        for event in result.get('events', []):
            if event['kind'] in ('contact_stop', 'drill_contact_dwell'):
                self._events.append({'timeSeconds': end, 'actionIndex': index, 'timing': 'action_boundary', 'event': copy.deepcopy(event)})
        return result


def capture_episode(scene_or_engine, actions=None, *, label='scripted', episode_id='episode',
                    interval_seconds=.2, expected_state_id=None, snapshot=None, protection=None, guard_config=None,
                    frame_collector=None):
    """Replay attempted actions, preserving an existing engine/snapshot guard contract.

    Pass a final engine or ``snapshot=...`` to verify the replay against its final
    state. If actions is omitted, accepted history is used; rejected attempts
    cannot be recovered from that history and are explicitly marked unknown.
    An optional collector observes the same replay through begin/observe/end
    hooks outside engine state; it must discard rejected actions and verify
    finalized outcomes before publishing any staged frames. Exceptions abort it.
    """
    if label not in LABELS:
        raise ValueError('label must be one of: ' + ', '.join(LABELS))
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError('interval_seconds must be finite and positive')
    baseline = scene_or_engine if isinstance(scene_or_engine, SurgicalEngine) else None
    scene = baseline.scene if baseline is not None else scene_or_engine
    if snapshot is not None:
        baseline = SurgicalEngine.restore(scene, snapshot)
    full_attempts = actions is not None
    if actions is None:
        if baseline is None:
            raise ValueError('Provide actions, a final engine, or a final snapshot')
        actions = baseline.actions
    if baseline is not None:
        expected_state_id = expected_state_id or baseline.state_id
        if protection is None:
            protection = getattr(baseline, 'protection', None)
        if guard_config is None:
            guard_config = getattr(baseline, 'guard_config', None)
    kwargs = {}
    if protection is not None:
        kwargs['protection'] = protection
    if guard_config is not None:
        kwargs['guard_config'] = guard_config
    engine = _ObservedEngine(scene, **kwargs)
    if frame_collector is not None:
        # The collector is shared by transactional clones via copy.copy;
        # staged observations never become patient state or engine identity.
        engine._frame_collector = frame_collector
    initial_id = engine.state_id
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError('Telemetry attempted actions must be dictionaries')
        engine.step(action)
    if expected_state_id is not None and engine.state_id != expected_state_id:
        raise ValueError('Telemetry replay state identity differs from the supplied final state')
    raw = engine._telemetry
    if not full_attempts:
        for sample in raw:
            sample['invalidActionsCumulative'] = None
    times = [sample['timeSeconds'] for sample in raw]
    end = times[-1]
    if end / interval_seconds > 1_000_000:
        raise ValueError('Reporting interval would exceed one million samples')
    ticks = [i * interval_seconds for i in range(math.floor((end + 1e-10) / interval_seconds) + 1)]
    if end - ticks[-1] > 1e-9:
        ticks.append(end)
    else:
        ticks[-1] = end
    rows = []
    for tick in ticks:
        index = max(0, bisect.bisect_right(times, tick + 1e-10) - 1)
        row = copy.deepcopy(raw[index])
        row.update(timeSeconds=tick, observedTimeSeconds=times[index],
                   observationAgeSeconds=max(0., tick - times[index]), observationIndex=index)
        rows.append(row)
    final = copy.deepcopy(raw[-1])
    return {'schema': SCHEMA, 'episodeId': str(episode_id), 'label': label, 'sceneId': scene.identity,
            'initialStateId': initial_id, 'finalStateId': engine.state_id,
            'expectedStateVerified': expected_state_id is not None,
            'attemptHistory': 'provided_attempts' if full_attempts else 'accepted_history_only_rejections_unknown',
            'intervalSeconds': interval_seconds, 'coordinateFrame': 'RAS+',
            'provenance': copy.deepcopy(PROVENANCE), 'reference': copy.deepcopy(REFERENCE),
            'physicsProvenance': copy.deepcopy(scene.properties.get('tissuePhysicsV2', scene.properties.get('tissuePhysics'))),
            'protection': engine.guard_report() if hasattr(engine, 'guard_report') else {'configured': False},
            'summary': {**{key: value for key, value in final.items() if key not in ('slots', 'sampleKind', 'actionIndex', 'kinematicIntervalSeconds')},
                        'pathLengthMm': dict(engine._path), 'attemptCount': len(engine._attempts),
                        'microstepCount': sum(sample['sampleKind'] == 'microstep' for sample in raw)},
            'attempts': engine._attempts, 'events': engine._events,
            'observations': raw, 'samples': rows}


def _flatten(episode, sample):
    row = {'episodeId': episode['episodeId'], 'label': episode['label'], 'sceneId': episode['sceneId'],
           'attemptHistory': episode['attemptHistory']}
    for key, value in sample.items():
        if key == 'slots':
            for hand, fields in value.items():
                for field, item in fields.items():
                    if isinstance(item, list):
                        suffixes = ('x', 'y', 'z', 'w')
                        row.update({f'{hand}_{field}_{suffixes[i]}': coordinate for i, coordinate in enumerate(item)})
                    elif item is None and field in ('velocityMmPerSecond', 'accelerationMmPerSecond2'):
                        row.update({f'{hand}_{field}_{axis}': None for axis in ('x', 'y', 'z')})
                    else:
                        row[f'{hand}_{field}'] = item
        else:
            row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    return row


def export_telemetry(episodes, output_dir):
    """Write inspectable JSON, reporting-grid CSV, and a self-contained timeline."""
    episodes = list(episodes)
    if not episodes or any(episode.get('schema') != SCHEMA for episode in episodes):
        raise ValueError('Provide at least one captured telemetry episode')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {kind: output / f'telemetry.{kind}' for kind in ('json', 'csv', 'html')}
    payload = {'schema': SCHEMA, 'reference': REFERENCE, 'provenance': PROVENANCE, 'episodes': episodes}
    paths['json'].write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    rows = [_flatten(episode, sample) for episode in episodes for sample in episode['samples']]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with paths['csv'].open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    paths['html'].write_text(_timeline_html(payload))
    return {kind: str(path.resolve()) for kind, path in paths.items()}


def _timeline_html(payload):
    # No remote scripts, fonts, or third-party code. Escape script delimiters in
    # user-controlled episode names and action reasons before embedding JSON.
    data = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    reference = html.escape(REFERENCE['citation'])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bimanual rehearsal telemetry</title><style>
body{font:15px system-ui,sans-serif;background:#f4f6f7;color:#172d39;margin:0;padding:28px;line-height:1.5}main{max-width:1240px;margin:auto}h1{font-size:30px;margin:0}h2{font-size:19px}p{max-width:1000px}.muted{color:#526571}section{background:white;border:1px solid #dae2e6;border-radius:12px;padding:20px;margin:18px 0}select{font:inherit;padding:8px;margin-right:18px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e2e7eb;padding:8px;text-align:left;vertical-align:top}th{white-space:nowrap}.scroll{overflow:auto;max-height:350px}svg{width:100%;height:260px;background:#fafcfd}code{font-size:12px}a{color:#1a6480}details{margin-top:15px}.legend{display:flex;gap:20px}.left{color:#087f8c}.right{color:#b64c25}
</style><main><h1>Bimanual rehearsal telemetry</h1><p class="muted">Time-resolved simulator observations. Motion and removal are model-derived; forces are assumed inputs. Clinical injury, patient risk, and expertise remain unknown.</p>
<section><h2>Episode comparison</h2><p>Labels identify how actions were generated. Episodes on different scenes, guard contracts, or durations are not controlled performance comparisons.</p><div class="scroll"><table id="comparison"></table></div></section>
<section><label>Episode <select id="episode"></select></label><label>Trace <select id="metric"><option value="speed">Tool speed (mm/s)</option><option value="acceleration">Tool acceleration magnitude (mm/s²)</option><option value="path">Tool path length (mm)</option><option value="active">Effective activation (0 / 1)</option><option value="removal">Removed volume (mm³)</option><option value="fluid">Fluid volume (mm³)</option><option value="separation">Tip separation (mm)</option><option value="wounds">Modeled wounds (count)</option></select></label>
<p id="detail" class="muted"></p><div id="legend" class="legend"></div><svg id="plot" viewBox="0 0 1100 260" role="img" aria-label="Selected telemetry over simulated time"></svg><p class="muted">Step traces use the last observed state. Null values leave gaps. The final row may be shorter than the nominal interval. Placement jumps are excluded from traveled path and derivatives.</p></section>
<section><h2>Action outcomes</h2><div class="scroll"><table id="actions"></table></div><h2>Selected reporting samples</h2><p class="muted">At most 200 rows shown here; the JSON and CSV retain every reporting row. JSON also retains all committed microstep observations and timed events.</p><div class="scroll"><table id="samples"></table></div></section>
<section><h2>Provenance and reference</h2><p>''' + reference + ''' <a href="https://doi.org/10.1038/s41746-022-00596-8">Paper</a>. Referenced for bimanual feature selection and 0.2-second reporting. No expertise model, thresholds, or external code were transferred.</p><details><summary>Field interpretation and limitations</summary><table id="provenance"></table></details></section></main>
<script id="data" type="application/json">''' + data + '''</script><script>
const D=JSON.parse(document.getElementById('data').textContent), E=D.episodes;
const el=id=>document.getElementById(id), fmt=x=>x==null?'unknown':typeof x==='number'?Number(x.toFixed(4)).toString():String(x);
function table(node,heads,rows){node.replaceChildren();const tr=document.createElement('tr');for(const x of heads){const th=document.createElement('th');th.textContent=x;tr.append(th)}node.append(tr);for(const row of rows){const tr=document.createElement('tr');for(const x of row){const td=document.createElement('td');td.textContent=fmt(x);tr.append(td)}node.append(tr)}}
table(el('comparison'),['Episode / label','Seconds','Path L / R (mm)','Target / non-target (mm³)','Invalid attempts','Exchanges'],E.map(e=>[e.episodeId+' / '+e.label,e.summary.timeSeconds,fmt(e.summary.pathLengthMm.left)+' / '+fmt(e.summary.pathLengthMm.right),fmt(e.summary.targetRemovedMm3)+' / '+fmt(e.summary.nonTargetRemovedMm3),e.attemptHistory==='provided_attempts'?e.summary.invalidActionsCumulative:'unknown (accepted history only)',e.summary.toolExchangesCumulative]));
E.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.episodeId+' — '+e.label;el('episode').append(o)});
table(el('provenance'),['Field','Interpretation'],Object.entries(D.provenance));
const fields={speed:[['Left',s=>s.slots.left.speedMmPerSecond],['Right',s=>s.slots.right.speedMmPerSecond]],acceleration:[['Left',s=>s.slots.left.accelerationMagnitudeMmPerSecond2],['Right',s=>s.slots.right.accelerationMagnitudeMmPerSecond2]],path:[['Left',s=>s.slots.left.pathLengthMm],['Right',s=>s.slots.right.pathLengthMm]],active:[['Left',s=>+s.slots.left.active],['Right',s=>+s.slots.right.active]],removal:[['Target',s=>s.targetRemovedMm3],['Non-target',s=>s.nonTargetRemovedMm3]],fluid:[['Present incl. outlet reservoirs',s=>s.fluidPresentMm3],['Removed',s=>s.fluidRemovedMm3]],separation:[['Tip separation',s=>s.tipSeparationMm]],wounds:[['Modeled wounds',s=>s.woundCount],['Active wounds',s=>s.activeWoundCount]]};
function draw(){const e=E[+el('episode').value],series=fields[el('metric').value],samples=e.samples,svg=el('plot');svg.replaceChildren();el('legend').replaceChildren();el('detail').textContent='Interval '+e.intervalSeconds+' s · '+e.summary.microstepCount+' committed microsteps · '+e.attemptHistory+' · max observation age '+fmt(samples.reduce((m,s)=>Math.max(m,s.observationAgeSeconds),0))+' s · final state '+e.finalStateId;
const ymax=series.reduce((m,[,f])=>samples.reduce((a,s)=>Math.max(a,f(s)||0),m),1), xmax=Math.max(.2,e.summary.timeSeconds), x=t=>60+t/xmax*1000,y=v=>220-v/ymax*185;
function node(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n}
node('path',{d:'M60 25V220H1060',stroke:'#9cabb3',fill:'none'});for(let i=0;i<=4;i++){node('text',{x:50,y:y(ymax*i/4)+4,'text-anchor':'end','font-size':11},fmt(ymax*i/4));node('text',{x:x(xmax*i/4),y:242,'text-anchor':'middle','font-size':11},fmt(xmax*i/4)+' s')}
series.forEach(([name,f],i)=>{const color=['#087f8c','#b64c25'][i];let d='',prev=null;for(const s of samples){const v=f(s);if(v===null){prev=null;continue}d+=(prev===null?'M'+x(s.timeSeconds)+' '+y(v):'H'+x(s.timeSeconds)+'V'+y(v));prev=v}node('path',{d,stroke:color,'stroke-width':2,fill:'none'});const span=document.createElement('span');span.style.color=color;span.textContent=name;el('legend').append(span)});
table(el('actions'),['#','Start / end (s)','Hand / operation','Status','Reason'],e.attempts.map(a=>[a.actionIndex,fmt(a.startTimeSeconds)+' / '+fmt(a.endTimeSeconds),(a.action.slot||'?')+' / '+(a.action.operation||'?'),a.status,a.reason||'—']));
table(el('samples'),['Time (s)','Observed (s)','Age (s)','L / R active','Separation (mm)','Target (mm³)','Non-target (mm³)','Fluid (mm³)','Contact (cell·s)'],samples.slice(0,200).map(s=>[s.timeSeconds,s.observedTimeSeconds,s.observationAgeSeconds,Number(s.slots.left.active)+' / '+Number(s.slots.right.active),s.tipSeparationMm,s.targetRemovedMm3,s.nonTargetRemovedMm3,s.fluidPresentMm3,s.contactCellSeconds]));}
el('episode').addEventListener('change',draw);el('metric').addEventListener('change',draw);draw();
</script></html>'''
