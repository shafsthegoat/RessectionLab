import {useEffect, useRef, useState} from 'react';
import {ReplayStepControl} from './ReplayStepControl';
import {checkedFamilyAvailability, requireInteractiveFamilyRequest} from './contact-family-availability';
import type {ContactFamilyAvailability, ContactFamilyMethod, ContactFamilyGoal, ContactFamilyRequest} from './contact-family-types';
import type {ContactFamilyView} from './contact-family-data';
import type {ResectionApi} from './types';
const METHODS: ContactFamilyMethod[] = ['STOP', 'SEARCH', 'IL', 'RL', 'IL_TRAIN_REFIT'];
const displayReason = (reason: string | null | undefined) => {
  if (reason === 'no_reviewed_final_32_update_pair_and_completed_pilot') return 'Final learned methods have not been released yet.';
  if (reason === 'backend_controller_not_promoted') return 'This generated task is not enabled in the current engine yet.';
  if (reason === 'backend_contact_release_failed_verification') return 'The learned release could not be verified. Inference is unavailable.';
  if (reason === 'backend_train_refit_release_failed_verification') return 'The TRAIN-only refit artifact is unavailable or could not be verified.';
  return reason;
};
const LABELS = {STOP: 'Immediate STOP', SEARCH: 'Bounded search', IL: 'Final imitation policy', RL: 'Final reinforcement policy', IL_TRAIN_REFIT: 'Full-teacher imitation · TRAIN only'};

export function ContactFamilyPanel({api, view, step, busy, visible, unavailableReason,
  onExecute, onStep, onShow, onSource, onFocusGoal}: {
  api: ResectionApi | null | undefined; view: ContactFamilyView | null; step: number; busy: boolean;
  visible: boolean; unavailableReason?: string;
  onExecute: (request: ContactFamilyRequest, catalog: ContactFamilyAvailability) => void;
  onStep: (step: number) => void; onShow: () => void; onSource: () => void; onFocusGoal: () => void;
}) {
  const [catalog, setCatalog] = useState<ContactFamilyAvailability | null>(null);
  const [pending, setPending] = useState(false), [error, setError] = useState<string | null>(null);
  const [layout, setLayout] = useState(''), [goal, setGoal] = useState<ContactFamilyGoal>('surface');
  const [method, setMethod] = useState<ContactFamilyMethod>('SEARCH');
  const generation = useRef(0);
  const refresh = async () => {
    const id = ++generation.current;
    setCatalog(null); setPending(true); setError(null);
    try {
      if (!api?.publicContactFamilyAvailability || api.readOnly || unavailableReason) throw Error(unavailableReason ?? 'This engine does not provide the family task.');
      const next = checkedFamilyAvailability(await api.publicContactFamilyAvailability({}));
      if (generation.current !== id) return;
      setCatalog(next);
      setLayout(current => next.layouts.some(row => row.layoutId === current && row.interactive) ? current : next.layouts.find(row => row.interactive)!.layoutId);
    } catch (cause) { if (generation.current === id) setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { if (generation.current === id) setPending(false); }
  };
  useEffect(() => {void refresh(); return () => {generation.current++;};}, [api, unavailableReason]);
  const selected = catalog?.layouts.find(row => row.layoutId === layout);
  const reason = unavailableReason ?? (!catalog ? 'Checking released methods…' : !selected?.interactive ? 'Held-out layouts are locked.' : method === 'IL_TRAIN_REFIT' && selected.role !== 'TRAIN' ? 'Full-teacher imitation is available only on TRAIN layouts.' : displayReason(catalog.methods[method]?.reason ?? (!catalog.methods[method] ? 'Method unavailable.' : null)));
  const execute = () => {
    if (!catalog || busy || pending || reason) return;
    try {
      const checked = requireInteractiveFamilyRequest({fixture: 'generated-public-contact-family-v2', layoutId: layout, goalId: goal, selector: method}, catalog);
      onExecute(checked.request, checked.catalog);
    } catch (cause) {setError(cause instanceof Error ? cause.message : String(cause));}
  };
  const episode = view?.episode, frame = episode?.replayFrames[step], prefix = view?.prefixes[step];
  return <section className="episode-panel" aria-label="Generated contact family">
    <div className="planning-title"><span className="eyebrow">GENERATED LAYOUTS</span><h2>Retain & touch · family</h2>
      <p>The same geometric contact objective on a declared generated layout. Choose a public goal and inspect the complete native replay.</p></div>
    <label className="field-label" htmlFor="family-layout">Layout and role</label>
    <select id="family-layout" value={layout} disabled={busy || pending || !catalog} onChange={event => setLayout(event.target.value)}>
      {!catalog && <option value="">Unavailable</option>}
      {catalog?.layouts.map(row => <option key={row.layoutId} value={row.layoutId} disabled={!row.interactive}>{row.layoutId} · {row.role}{!row.interactive ? ' · locked' : ''}</option>)}
    </select>
    <label className="field-label" htmlFor="family-goal">Public goal</label>
    <select id="family-goal" value={goal} disabled={busy || pending} onChange={event => setGoal(event.target.value as ContactFamilyGoal)}><option value="surface">Surface</option><option value="deep">Deep</option></select>
    <label className="field-label" htmlFor="family-method">Method</label>
    <select id="family-method" value={method} disabled={busy || pending || !catalog} onChange={event => setMethod(event.target.value as ContactFamilyMethod)}>
      {METHODS.map(value => <option key={value} value={value} disabled={!catalog?.methods[value]?.available || value === 'IL_TRAIN_REFIT' && selected?.role !== 'TRAIN'}>{LABELS[value]}{catalog && !catalog.methods[value]?.available ? ' · unavailable' : ''}</option>)}
    </select>
    <button className="primary-button" disabled={busy || pending || !!reason} onClick={execute}>{busy ? 'Executing / checking…' : 'Execute selected method'}</button>
    <button className="text-button" disabled={busy || pending} onClick={() => void refresh()}>Refresh availability</button>
    {error && <p role="alert">{error}</p>}{reason && <p className="muted-note" role="status">{reason}</p>}
    {catalog && !catalog.methods.IL.available && <p className="instrument-note">{displayReason(catalog.methods.IL.reason)} No untrained substitute is run.</p>}
    {method === 'IL_TRAIN_REFIT' && <p className="instrument-note">Full-teacher imitation uses extra TRAIN fitting: 1,280 loss evaluations versus 128 in the original pilot. Its fixed native TRAIN pass contacted 6 of 24 goals; saved bounded search contacted 16 of 24. Eighteen episodes chose STOP only. This is not held-out performance. Every TRAIN layout remains available for inspection.</p>}
    <p className="instrument-note">TRAIN and SELECT are interactive; held-out layouts stay locked. Learned methods use released final artifacts for inference only. This control starts no training. One episode establishes neither superiority nor generalization.</p>
    {view && episode && frame && prefix && <>
      <div className="episode-status" role="status"><strong>{view.policyVariant === 'IL_TRAIN_REFIT' ? 'Full-teacher imitation · TRAIN only · live owned inference' : episode.selector === 'IL' || episode.selector === 'RL' ? 'Final learned policy · live owned inference' : LABELS[episode.selector]}</strong>
        <span>{episode.layoutId} · {episode.splitRole} · {episode.publicGoal.goalId} goal · native replay checked</span></div>
      <p>Goal at native cell {episode.publicGoal.nativeIndex.join(', ')}; RAS+ mm {episode.publicGoal.rasMm.map(n => n.toFixed(1)).join(', ')}. The marker is a public task location, not anatomy.</p>
      <p className="instrument-note">Temporary episode; workspace saving, vessel evaluation and the old aspiration comparison are unavailable. All tissue removal is charged. The zero nominal-target field carries no reward.</p>
      <div className="episode-replay-buttons"><button disabled={busy} onClick={onFocusGoal}>Focus public goal</button><button disabled={busy || visible} onClick={onShow}>Show replay</button><button disabled={!visible} onClick={onSource}>Source view</button></div>
      <ReplayStepControl id="family-replay-step" requestedStep={step} appliedStep={step} stepCount={view.frames.length - 1} disabled={busy} onRequest={onStep}/>
      <p className="episode-pose"><strong>{frame.phase === 'initial' ? 'Initial tissue' : frame.phase === 'stop' ? 'STOP · committed terminal action' : `${frame.mode === 'probe' ? 'Probe contact' : 'Aspiration'} / ${frame.phase}`}</strong></p>
      <dl className="episode-quantities"><div><dt>Goal at this frame</dt><dd>{!prefix.retained ? 'Removed · no contact credit' : prefix.contactedAndRetained ? 'Retained and contacted' : 'Retained · not contacted'}</dd></div><div><dt>All removal at frame</dt><dd>{prefix.removedVolumeMm3.toFixed(2)} mm³</dd></div></dl>
      {episode.selector === 'SEARCH' && (episode.planning.time_cap_reached === true || episode.planning.call_cap_reached === true || Number(episode.planning.beam_pruned_prefixes) > 0) && <p className="instrument-note">Search was incomplete{episode.planning.time_cap_reached === true ? ' because it reached its time limit' : episode.planning.call_cap_reached === true ? ' because it reached its call limit' : ' because its beam pruned candidate prefixes'}. This is its returned bounded result, not an optimum.</p>}
      <h3>Complete native outcome</h3><dl className="episode-quantities"><div><dt>Modeled return</dt><dd>{view.outcome.modeledReturn.toFixed(3)}</dd></div><div><dt>Total removal</dt><dd>{view.outcome.totalRemovedMm3.toFixed(2)} mm³</dd></div><div><dt>Round-trip path</dt><dd>{view.outcome.completePathMm.toFixed(2)} mm</dd></div><div><dt>Effort and removal cost</dt><dd>{view.outcome.totalCost.toFixed(3)}</dd></div><div><dt>Termination</dt><dd>{view.outcome.termination === 'STOP' ? 'STOP' : 'Two-action limit'}</dd></div></dl>
      <ol className="episode-timeline">{episode.history.map((action, i) => <li key={i}><button disabled={busy} onClick={() => onStep(episode.replayFrames.findIndex(row => row.actionIndex === i))}>{i + 1}. {action.interaction_mode} · {action.removed_indices_native.length} cells removed · reward {action.reward.toFixed(3)}</button></li>)}</ol>
      <details className="episode-evidence"><summary>Layout, method & public-input provenance</summary>
        <p>{view.observation.cropShape.join(' × ')} public crop; structural, support, zero compatibility target, initial cavity, goal and committed contact are available. Motor and language channels are unavailable, not negative findings. No private reference was scored.</p>
        <p>Contact earns 1 only while the goal remains intact. Native removal, action, motion and tool-change costs determine the return. Geometric contact is not a force, sensor or clinical result.</p>
        {view.policyVariant === 'IL_TRAIN_REFIT' && <p>Separate full-teacher TRAIN refit: 32 updates using all 40 saved teacher states per update. Fixed native TRAIN outcomes were 6/24 contacts, 18 STOP-only episodes, versus 16/24 for saved bounded search. Original IL/RL artifacts remain separate; no held-out or clinical conclusion follows.</p>}
        {episode.learnedAuthorship ? <p>Final {episode.learnedAuthorship.method}: {episode.learnedAuthorship.completedUpdates} recorded training updates; this inference made {Number(episode.planning.actor_forward_calls)} actor calls and zero optimizer updates. Artifact and owned-run bindings are consistency evidence, not signed proof of training.</p> : <p>No policy inference. SEARCH reports its returned bounded result, without a global-optimum claim.</p>}
        <dl className="episode-identities"><dt>Episode</dt><dd>{episode.episodeId}</dd><dt>Family / source</dt><dd>{episode.familyHash}<br/>{episode.sourceHash}</dd><dt>Candidate proposals</dt><dd>{episode.taskContract.sourceCandidateVersion} · exact floating-point native paths</dd><dt>Public objective</dt><dd>{episode.publicGoal.objectiveHash}</dd>{episode.learnedAuthorship && <><dt>Checkpoint / parameters</dt><dd>{episode.learnedAuthorship.checkpointFileSha256}<br/>{episode.learnedAuthorship.parameterHash}</dd></>}</dl>
      </details>
    </>}
  </section>;
}
