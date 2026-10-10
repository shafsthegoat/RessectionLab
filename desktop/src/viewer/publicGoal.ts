import {rasAffine,transformPoint,PLANE_AXES,sliceRect} from './coordinates.ts';
import type {Bounds3,Point3,SlicePlane} from './coordinates.ts';
import type {ViewerVolume} from './contracts.ts';
export interface ViewerPublicGoal {scope:'declared-public-geometric-goal';caseHash:string;episodeId:string;objectiveHash:string;nativeIndex:Point3;rasMm:Point3;state:'retained-uncontacted'|'retained-contacted'|'removed'}
export const PUBLIC_GOAL_COLOR='#f1c75b';
export function checkedPublicGoal(volume:ViewerVolume,goal:ViewerPublicGoal|null):ViewerPublicGoal|null{
 if(!goal)return null;
 const hash=/^sha256:[a-f0-9]{64}$/;
 if(goal.scope!=='declared-public-geometric-goal'||goal.caseHash!==volume.caseHash||![goal.episodeId,goal.objectiveHash].every(h=>hash.test(h))||!['retained-uncontacted','retained-contacted','removed'].includes(goal.state)||goal.nativeIndex.length!==3||goal.nativeIndex.some((n,i)=>!Number.isSafeInteger(n)||n<0||n>=volume.shape[i])||goal.rasMm.length!==3)throw Error('Public goal source/identity changed');
 const ras=transformPoint(rasAffine(volume.affine,volume.frame),goal.nativeIndex);if(ras.some((n,i)=>!Number.isFinite(goal.rasMm[i])||Math.abs(n-goal.rasMm[i])>1e-9))throw Error('Public goal RAS/native coordinates disagree');
 return {...goal,nativeIndex:[...goal.nativeIndex],rasMm:[...goal.rasMm]};
}
export function publicGoalSlicePoint(goal:ViewerPublicGoal,bounds:Bounds3,plane:SlicePlane,cursor:Point3,width:number,height:number){
 const[a,b,c]=PLANE_AXES[plane];if(Math.abs(goal.rasMm[c]-cursor[c])>1e-6)return null;
 const r=sliceRect(bounds,plane,width,height);return {left:r.left+(goal.rasMm[a]-bounds[0][a])/(bounds[1][a]-bounds[0][a])*r.width,top:r.top+(bounds[1][b]-goal.rasMm[b])/(bounds[1][b]-bounds[0][b])*r.height};
}
