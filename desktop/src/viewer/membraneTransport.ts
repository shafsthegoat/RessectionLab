import type {SurgicalView} from '../surgical-types';
import type {Point3} from './coordinates';

const amount=(n:number)=>Number(n.toFixed(2)).toString();

/** Only describe the patch owned by the displayed episode's assessed
 * stage. Rigid offset is checked against every retained source membership. */
export function membraneTransport(view:SurgicalView):string|null {
  const stage=view.cranialStages;
  if(!stage||stage.schema!=='resectionlab.cranial-stages.v1'||stage.sceneId!==view.scene.sceneId||
    stage.stateId!==view.stateId||!stage.assessmentCurrent||!stage.bonePrefixVerified)return null;
  if(stage.patchExtracted&&stage.duraCleared&&stage.membrane?.actualNativeExtractionEvents===1){
    const volume=stage.membrane.removedMembraneMm3;
    return Number.isFinite(volume)?`Membrane extracted · ${amount(volume)} mm³`:null;
  }
  if(stage.patchDetached)return 'Membrane released in field';
  if(!stage.patchHeld){
    if(stage.boundaryCutComplete)return 'Membrane boundary cut · patch retained';
    return stage.cutBoundaryEdges>0?`Membrane cut · ${stage.cutBoundaryEdges} of ${stage.requiredBoundaryEdges} edges`:null;
  }
  const held=Object.entries(view.holds??{}).find(([hand,hold])=>
    view.slots[hand]?.instrument==='membrane_forceps'&&'kind' in hold&&hold.kind==='membrane-patch-v1'&&
    'contractHash' in hold&&hold.contractHash===stage.contractHash);
  if(!held)return 'Membrane held';
  const [hand,hold]=held;
  if(!('offset' in hold)||!Array.isArray(hold.offset)||hold.offset.length!==3||!hold.offset.every(Number.isSafeInteger)||
    !hold.cells.length||hold.cells.length!==stage.patchCells)return 'Membrane held';
  const delta=hold.offset as Point3;
  const [nx,ny,nz]=view.scene.shape,moved=new Map(view.moved),removed=new Set(view.removed);
  if(!hold.cells.every(cell=>{
    if(!Number.isSafeInteger(cell)||cell<0||cell>=nx*ny*nz||removed.has(cell))return false;
    const x=Math.floor(cell/(ny*nz))+delta[0],y=Math.floor(cell/nz)%ny+delta[1],z=cell%nz+delta[2];
    return x>=0&&y>=0&&z>=0&&x<nx&&y<ny&&z<nz&&(moved.get(cell)??cell)===(x*ny+y)*nz+z;
  }))return 'Membrane held';
  const mm=Math.hypot(...view.scene.affine.slice(0,3).map(row=>row[0]*delta[0]+row[1]*delta[1]+row[2]*delta[2]));
  if(!Number.isFinite(mm))return 'Membrane held';
  return mm>0?`Membrane held · ${amount(mm)} mm from source · ${hand} hand`:`Membrane held at source · ${hand} hand`;
}
