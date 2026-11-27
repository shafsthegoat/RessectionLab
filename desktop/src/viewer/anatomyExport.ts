import * as THREE from 'three';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { GpuFrameTimer } from './gpuFrameTimer';
import { renderTargetBytes } from './renderResourceEstimate';
import { ScanVolumeExportTargets, type ScanVolumeRenderer } from './ScanVolumeRenderer';
import { renderAnatomyScene, type VesselSlabContext } from './vesselSlabContext';

export const ANATOMY_PNG_LONG_EDGE = 2048;
const SCAN_SAMPLES = 32;
const MAX_ATTACHMENT_BYTES = 160 * 1024 ** 2;
// 128 bytes per pixel covers the opaque, ray, accumulation, peel, cranial
// depth and output attachments even when every surface group is present.
const MAX_TILE_PIXELS = Math.floor(MAX_ATTACHMENT_BYTES / 128);
// The cortex depth shader samples up to 12 pixels away.
const GUARD_ROWS = 16;

/** Horizontal bands share the same full projection, with overlapping rows for screen-space shading. */
export function anatomyExportBands(aspect: number, longEdge: number) {
  const width = Math.max(1,Math.round(longEdge * Math.min(aspect,1)));
  const height = Math.max(1,Math.round(longEdge / Math.max(aspect,1)));
  const bandRows = Math.floor(MAX_TILE_PIXELS / width) - 2 * GUARD_ROWS;
  const bands = [];
  for (let row = 0; row < height; row += bandRows) {
    const rows = Math.min(bandRows,height - row);
    const top = Math.max(0,row - GUARD_ROWS);
    const bottom = Math.min(height,row + rows + GUARD_ROWS);
    bands.push({row, rows, top, height:bottom - top});
  }
  return {width,height,bands};
}

/** WebGL orders rows from the bottom and the PNG surface orders them from the top. */
export function copyAnatomyExportBand(pixels: Uint8Array, image: ImageData, band: {row:number; rows:number; top:number; height:number}) {
  const stride = image.width * 4;
  for (let row = band.row; row < band.row + band.rows; row++) {
    const start = (band.height - 1 - (row - band.top)) * stride;
    image.data.set(pixels.subarray(start,start + stride),row * stride);
  }
}

function encodePng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve,reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob);
    else reject(new Error('The 3D image could not be encoded as PNG.'));
  },'image/png'));
}

async function waitForGpu(gl: WebGL2RenderingContext, signal: AbortSignal) {
  const started = performance.now();
  const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
  if (!fence) throw new Error('The GPU could not submit the 3D export.');
  try {
    gl.flush();
    for (;;) {
      // Yield between samples so cancellation and other desktop work can run.
      await new Promise(resolve => setTimeout(resolve,8));
      signal.throwIfAborted();
      const status = gl.clientWaitSync(fence,0,0);
      if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) return performance.now() - started;
      if (status !== gl.TIMEOUT_EXPIRED || performance.now() - started > 60000)
        throw new Error('The GPU did not finish a sample of the 3D export.');
    }
  } finally {
    gl.deleteSync(fence);
  }
}

export async function renderAnatomyPng(
  renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera,
  scan: ScanVolumeRenderer | null, signal: AbortSignal, slabContext?:VesselSlabContext,
) {
  const started = performance.now();
  signal.throwIfAborted();
  const gl = renderer.getContext() as WebGL2RenderingContext;
  const longEdge = Math.min(ANATOMY_PNG_LONG_EDGE,renderer.capabilities.maxTextureSize,gl.getParameter(gl.MAX_RENDERBUFFER_SIZE));
  if (gl.isContextLost()) throw new Error('Graphics must be available to export the 3D view.');
  const {width,height,bands} = anatomyExportBands(camera.aspect,longEdge);
  const canvas = document.createElement('canvas');
  canvas.width=width;
  canvas.height=height;
  const context = canvas.getContext('2d',{alpha:false});
  if (!context) throw new Error('The 3D export could not allocate its PNG image.');
  const image = context.createImageData(width,height);
  const samples = scan?.active ? SCAN_SAMPLES : 1;
  const tileCamera = camera.clone();
  const depthPass = scan ? new ScanVolumeExportTargets() : null;
  const meshTarget = scan ? null : new THREE.WebGLRenderTarget(1,1,{
    type:THREE.HalfFloatType, samples:Math.min(4,renderer.capabilities.maxSamples),
  });
  const outputTarget = new THREE.WebGLRenderTarget(1,1,{depthBuffer:false});
  const outputPass = new OutputPass();
  const pixels = new Uint8Array(width * Math.max(...bands.map(band=>band.height)) * 4);
  let cpuSubmissionMs = 0, completionPollMs = 0, readbackMs = 0, peakAttachmentBytes = 0;
  let gpuElapsedMs = 0, gpuTimedSamples = 0, serial = 0;
  const gpuTimer = new GpuFrameTimer(gl,sample => {
    if(!sample.completed || sample.milliseconds === undefined) return;
    gpuElapsedMs += sample.milliseconds;
    gpuTimedSamples++;
  }, callback => window.setTimeout(()=>callback(performance.now()),8), handle => window.clearTimeout(handle));
  try {
    for (const band of bands) {
      tileCamera.setViewOffset(width,height,0,band.top,width,band.height);
      outputTarget.setSize(width,band.height);
      meshTarget?.setSize(width,band.height);
      for (let sample = 0; sample < samples; sample++) {
        signal.throwIfAborted();
        const submitted = performance.now();
        const previousTarget = renderer.getRenderTarget();
        const onShaderError = renderer.debug.onShaderError;
        renderer.debug.onShaderError = () => { throw new Error('The GPU could not compile the 3D export.'); };
        gpuTimer.begin(++serial);
        try {
          if (meshTarget) {
            renderer.setRenderTarget(meshTarget);
            renderer.clear();
            renderAnatomyScene(renderer,scene,tileCamera,slabContext);
          }
          const linear = scan && depthPass
            ? scan.renderExportSample(scene,tileCamera,depthPass,width,band.height,sample,height - band.top - band.height,slabContext)
            : meshTarget!;
          if (sample === samples - 1) outputPass.render(renderer,outputTarget,linear,0,false);
        } finally {
          gpuTimer.end();
          renderer.debug.onShaderError = onShaderError;
          renderer.setRenderTarget(previousTarget);
        }
        cpuSubmissionMs += performance.now() - submitted;
        completionPollMs += await waitForGpu(gl,signal);
        peakAttachmentBytes = Math.max(peakAttachmentBytes,renderTargetBytes(outputTarget) +
          (depthPass?.estimatedBytes ?? renderTargetBytes(meshTarget!)));
      }
      const reading = performance.now();
      renderer.readRenderTargetPixels(outputTarget,0,0,width,band.height,pixels);
      copyAnatomyExportBand(pixels,image,band);
      readbackMs += performance.now() - reading;
    }
    const encoding = performance.now();
    context.putImageData(image,0,0);
    const blob = await encodePng(canvas);
    signal.throwIfAborted();
    return {blob, metrics:{
      width,height,samples,bands:bands.length,bytes:blob.size,
      elapsedMs:performance.now() - started,cpuSubmissionMs,completionPollMs,readbackMs,
      encodingMs:performance.now() - encoding,peakAttachmentBytes,
      gpuTimingSupported:gpuTimer.supported,gpuElapsedMs:gpuTimedSamples ? gpuElapsedMs : null,gpuTimedSamples,
      ...(scan?.exportMode ? {exportMode:scan.exportMode} : {}),
    }};
  } finally {
    gpuTimer.dispose();
    outputPass.dispose();
    outputTarget.dispose();
    depthPass?.dispose();
    meshTarget?.dispose();
  }
}
