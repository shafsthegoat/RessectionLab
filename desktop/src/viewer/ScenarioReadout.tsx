import type {ScenarioLine} from '../operative-scenarios';

/** Scenario, estimate and rehearsal-truth lines inside the existing operative readout. */
export function ScenarioReadout({lines}:{lines:readonly ScenarioLine[]}) {
  if(!lines.length)return null;
  return <div className="operative-scenarios" aria-label="Operative scenarios">
    {lines.map(line=><span key={line.key} data-tone={line.tone}>{line.text}</span>)}
  </div>;
}
