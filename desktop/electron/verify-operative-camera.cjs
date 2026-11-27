'use strict';

// Serialized into a hidden renderer. Three.js publishes new renderers to this
// optional observer before the field scene starts submitting camera matrices.
function observeField() {
  const probe = { last: null, pinned: null, lastDraw: null, cameras: {} };
  window.__operativeQA = probe;
  const events = window.__THREE_DEVTOOLS__ ??= new EventTarget();
  events.addEventListener('observe', ({ detail: renderer }) => {
    if (!renderer.isWebGLRenderer) return;
    const render = renderer.render.bind(renderer);
    renderer.render = (scene, camera) => {
      const host = renderer.domElement.closest('.operative-field-canvas');
      if (!host || !(camera.isOrthographicCamera || camera.isPerspectiveCamera)) return render(scene, camera);
      const projection = camera.isOrthographicCamera ? 'orthographic' : 'perspective';
      probe.cameras[projection] = camera;
      probe.last = { renderer, scene, camera };
      let drawn = camera;
      if (probe.pinned) {
        const type = probe.pinned.projection ?? 'orthographic';
        if (!probe.cameras[type]) throw Error('Missing '+type+' comparison camera');
        drawn = probe.cameras[type].clone();
        drawn.matrixWorld.fromArray(probe.pinned.matrixWorld);
        drawn.matrixWorld.decompose(drawn.position, drawn.quaternion, drawn.scale);
        drawn.matrix.copy(drawn.matrixWorld);
        drawn.matrixAutoUpdate = false;
        drawn.matrixWorldAutoUpdate = false;
        drawn.matrixWorldInverse.copy(drawn.matrixWorld).invert();
        drawn.projectionMatrix.fromArray(probe.pinned.projectionMatrix);
        drawn.projectionMatrixInverse.copy(drawn.projectionMatrix).invert();
      }
      const sampling=probe.sampling, gl=sampling&&renderer.getContext();
      const query=sampling?.timer && !gl.getQuery(sampling.timer.TIME_ELAPSED_EXT,gl.CURRENT_QUERY) ? gl.createQuery() : null;
      if(query)gl.beginQuery(sampling.timer.TIME_ELAPSED_EXT,query);
      const start=sampling?performance.now():0;
      render(scene, drawn);
      if(sampling){
        const submitted=performance.now();
        if(query)gl.endQuery(sampling.timer.TIME_ELAPSED_EXT);
        sampling.frames.push({start,cpuMs:submitted-start,query});
      }
      const tool = scene.getObjectByName('hollow-aspirator');
      let tipViewport = null;
      if (tool?.visible) {
        const tip = tool.position.clone().set(0, 0, 0);
        tool.localToWorld(tip).project(drawn);
        tipViewport = [(tip.x + 1) / 2, (1 - tip.y) / 2];
      }
      probe.lastDraw = {
        projection: drawn.isOrthographicCamera ? 'orthographic' : 'perspective',
        tipViewport,
        canvas: renderer.domElement.getBoundingClientRect().toJSON(),
        matrixWorld: drawn.matrixWorld.toArray(),
        projectionMatrix: drawn.projectionMatrix.toArray(),
        drawCalls: renderer.info.render.calls,
        triangles: renderer.info.render.triangles,
      };
    };
  });
  probe.layers = () => {
    const result = [];
    probe.last.scene.traverseVisible(mesh => {
      if (!mesh.isMesh || !mesh.name) return;
      const geometry = mesh.geometry;
      result.push({
        name: mesh.name,
        vertices: geometry.attributes.position?.count,
        triangles: (geometry.index?.count ?? geometry.attributes.position?.count ?? 0) / 3,
        material: [mesh.material].flat().map(m => ({
          name: m.name, color: m.color?.getHexString(), opacity: m.opacity,
          roughness: m.roughness, metalness: m.metalness,
        })),
      });
    });
    return result;
  };
  probe.paint = pose => {
    probe.pinned = pose;
    const { renderer, scene, camera } = probe.last;
    renderer.render(scene, camera);
  };
}
module.exports = { observeField };
