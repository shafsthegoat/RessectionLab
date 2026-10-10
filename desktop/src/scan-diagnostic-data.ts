import type {ArrayDescriptor,ResectionApi} from './types.ts';
import type {DisplaySeriesView} from './workspace-imaging-types.ts';
import type {UnreviewedCase4DiagnosticLayerV1} from './scan-diagnostic-layer.ts';
import type {LoadedDiagnosticLayer} from './viewer/diagnosticLayer.ts';
import {validateDiagnosticLayer} from './viewer/diagnosticLayer.ts';
import {sha256Bytes} from './source-integrity.ts';
import {validateDiagnosticResult} from './scan-diagnostic-contract.ts';

export interface DiagnosticRequest {caseHash:string;seriesId:string}
export interface DiagnosticTransport {
  caseHash:string;seriesId:string;descriptor:UnreviewedCase4DiagnosticLayerV1;sourceSha256:string;
  stateGrid:LoadedDiagnosticLayer['stateGrid'];coverageGrid:LoadedDiagnosticLayer['coverageGrid'];
  loadedStateSha256:string;loadedCoverageSha256:string;state:ArrayDescriptor;coverage:ArrayDescriptor;
}
export interface DiagnosticView {source:DisplaySeriesView;layer:LoadedDiagnosticLayer;sourceSha256:string}
export async function hydrateDiagnostic(input:DiagnosticTransport,selected:DisplaySeriesView,api:ResectionApi):Promise<DiagnosticView>{
  const value=structuredClone(input);
  const request={caseHash:selected.descriptor.referenceCaseHash,seriesId:selected.descriptor.seriesId};
  validateDiagnosticResult(value,request);
  const refs=selected.descriptor.volume.sourceRefs?.filter(r=>r.source_id==='structural'&&r.provenance==='observed')??[];
  if(refs.length!==1||refs[0].sha256!==value.sourceSha256)throw Error('Select the exact prepared T1c image for this diagnostic.');
  if(JSON.stringify(value.descriptor.shape_xyz)!==JSON.stringify(selected.volume.shape))throw Error('Diagnostic grid differs from the displayed image.');
  const read=async(d:ArrayDescriptor)=>{const bytes=new Uint8Array(await api.readAsset(d.assetId)).slice();if(bytes.byteLength!==d.byteLength||await sha256Bytes(bytes)!==d.sha256)throw Error('Diagnostic transfer checksum changed.');return bytes};
  const [state,coverage]=await Promise.all([read(value.state),read(value.coverage)]);
  const layer:LoadedDiagnosticLayer={descriptor:value.descriptor,stateXYZ:new Int8Array(state.buffer),coverageXYZ:coverage,
    stateGrid:value.stateGrid,coverageGrid:value.coverageGrid,loadedStateSha256:value.loadedStateSha256,loadedCoverageSha256:value.loadedCoverageSha256,
    primaryOverlaysPermitted:false,planningInteractionPermitted:false};
  validateDiagnosticLayer(layer,selected.volume,value.sourceSha256);
  return {source:selected,layer,sourceSha256:value.sourceSha256};
}
