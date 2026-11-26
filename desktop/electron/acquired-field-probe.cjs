'use strict';

// The geometric field has its own renderer and no surgical state ID. The
// timeline may update before React applies the corresponding mesh.
function installFieldProbe() {
  const data = window.acquiredField = {
    current: null,
    last: null,
    sample: null,
    start(position, seek) {
      const timeline = document.querySelector('[aria-label="Operative timeline"]');
      if (this.current || !this.last || Number(timeline.value) === position)
        throw Error('Choose a different, settled geometric phase');
      this.sample = null;
      this.current = {position, previousGeometry: this.last.geometry, started: performance.now()};
      if (Number(seek()) !== position) throw Error('The operative timeline rejected the position');
    },
    visibleMeshes() {
      const meshes = [];
      this.last.scene.traverseVisible(mesh => {
        if (!mesh.isMesh || !mesh.name) return;
        const {attributes, index} = mesh.geometry;
        const vertices = attributes.position?.count ?? 0;
        meshes.push({name: mesh.name, vertices, triangles: (index?.count ?? vertices) / 3});
      });
      return meshes;
    }
  };
  // As in the shared operative camera verifier, observe Three.js renderers
  // before creating the field. The render parameters remain the app's own.
  const events = window.__THREE_DEVTOOLS__ ??= new EventTarget();
  events.addEventListener('observe', ({detail: renderer}) => {
    if (!renderer.isWebGLRenderer) return;
    const render = renderer.render.bind(renderer);
    renderer.render = (scene, camera) => {
      const canvas = renderer.domElement;
      const host = canvas.closest('.operative-field-canvas');
      const geometry = host && scene.getObjectByName('tumor')?.geometry;
      if (!geometry) return render(scene, camera);
      const playback = host.closest('.operative-playback');
      const timeline = playback.querySelector('[aria-label="Operative timeline"]');
      const request = data.current;
      const match = request && !request.fenced && request.previousGeometry !== geometry &&
        request.position === Number(timeline.value) && playback.dataset.operativePhase === 'tumor';
      const drawStarted = match ? performance.now() : 0;
      render(scene, camera);
      data.last = {geometry, scene};
      if (!match) return;
      request.fenced = true;
      const submitted = performance.now();
      const {calls: drawCalls, triangles} = renderer.info.render;
      const gl = renderer.getContext();
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      if (!sync) throw Error('The acquired geometric phase could not be fenced');
      gl.flush();
      function poll() {
        const code = gl.clientWaitSync(sync, 0, 0);
        if (code === gl.TIMEOUT_EXPIRED) {
          requestAnimationFrame(poll);
          return;
        }
        gl.deleteSync(sync);
        if (code === gl.WAIT_FAILED) throw Error('Geometric phase GPU fence failed');
        const completed = performance.now();
        const vertices = geometry.attributes.position.count;
        data.sample = {
          position: request.position, phase: playback.dataset.operativePhase,
          progress: playback.dataset.operativeProgress, pattern: host.dataset.aspirationPattern,
          surface: host.dataset.corticalSurface, camera: JSON.parse(canvas.dataset.camera),
          canvasPixels: [canvas.width, canvas.height], triangles, drawCalls,
          targetMesh: {vertices, triangles: (geometry.index?.count ?? vertices) / 3},
          inputToCpuSubmissionMs: submitted - request.started, inputToGpuMs: completed - request.started,
          cpuSubmissionMs: submitted - drawStarted, submissionToGpuMs: completed - submitted
        };
        data.current = null;
      }
      requestAnimationFrame(poll);
    };
  });
}

module.exports = {installFieldProbe};
