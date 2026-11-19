import type {NativeWorkspace} from './use-native-workspace';
import type {CranialView} from './native-anatomy-data';

export function NativeCranialView({workspace:w}:{workspace:NativeWorkspace}){
  if(!w.hasCranialView)return null;
  const canPlan=!w.planning.targetSelectionIssue&&w.planning.targetOptions.some(item=>item.layerId===w.planning.activeTargetLayerId&&item.target);
  return <label className="unified-source">Cranial view
    <select aria-label="Cranial view" value={w.state.cranialView??'custom'} disabled={w.busy||!!w.draft||w.planning.pending}
      onChange={event=>w.composeCranialView(event.target.value as CranialView|'custom'|'restore')}>
      <option value="custom">Custom layers · all approaches</option>
      <option value="planning" disabled={!canPlan}>Cortex and target · planning</option>
      <option value="inspection">Internal boundaries · inspection</option>
      {w.canRestoreCranialView&&<option value="restore">Restore previous view</option>}
    </select>
  </label>;
}
