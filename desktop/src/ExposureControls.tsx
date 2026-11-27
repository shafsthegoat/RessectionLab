import {useState} from 'react';
import type {OperativeExposure, ExposureBodyName} from './operative-exposure-types';
import type {Point3} from './viewer/coordinates';

const bodyNames: ExposureBodyName[] = ['scalp', 'bone', 'dura'];
const titles = {scalp: 'Scalp and galea', bone: 'Calvarial flap', dura: 'Dural flap'};
const showPoint = (point: Point3) => point.map(value => Number(value.toFixed(1))).join(', ');
const describe = (value: string) => value.replace(/[-_]/g, ' ');

export function ExposureControls({exposure, draft}: {
  exposure: OperativeExposure;
  draft?: (point: Point3) => void;
}) {
  const [selected, select] = useState<ExposureBodyName>('scalp');
  const body = exposure.bodies[selected];
  const carried = exposure.objects.filter(item => item.status === 'carried').length;
  const retrieved = exposure.objects.filter(item => item.status === 'retrieved').length;
  const placed = exposure.objects.length - carried - retrieved;
  return <section className="exposure-controls" aria-label="Operative exposure">
    <header><h3>Operative exposure</h3>
      <span>{exposure.complete ? 'Closure complete' : exposure.accessOpen ? 'Open surgical field' : 'Layered exposure'}</span>
    </header>
    <div className="exposure-layers" role="group" aria-label="Inspect a tissue layer">
      {bodyNames.map(name => <button key={name} aria-pressed={name === selected} onClick={() => select(name)}>
        <strong>{titles[name]}</strong><small>{describe(exposure.bodies[name].status)}</small>
      </button>)}
    </div>
    <p>{body.cutCount} of {body.edgeCount} perimeter attachments divided
      {body.heldBy && <> · Grasped by {body.heldBy} hand</>}
      {selected !== 'bone' && <> · Fold {Math.round(180 * body.angleRad / Math.PI)}°</>}
      {body.closedAnchors.length > 0 && <> · {body.closedAnchors.length} closure points secured</>}
    </p>
    <details><summary>Grip and closure landmarks</summary>
      <p className="surgical-note">Source-bound positions in RAS millimeters. Selecting a point sets the proposed tool tip.</p>
      <div className="surgical-actions">
        {[{label:'Grip', point:body.gripMm}, ...body.anchorsMm.map((point, i) => ({label:`Anchor ${i + 1}`, point}))].map(({label, point}) =>
          <button key={label} disabled={!draft} onClick={() => draft?.(point)}>{label} · {showPoint(point)}</button>)}
      </div>
    </details>
    {exposure.objects.length > 0 && <details>
      <summary>Field count · {placed} in place · {carried} carried · {retrieved} retrieved</summary>
      <ul>{exposure.objects.map(item => <li key={item.id}>{describe(item.kind)} · {item.body} · {describe(item.status)}</li>)}</ul>
    </details>}
  </section>;
}
