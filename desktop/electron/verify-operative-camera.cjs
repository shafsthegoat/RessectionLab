'use strict';

// Serialized into a hidden renderer. Three.js publishes new renderers to this
// optional observer before the field scene starts submitting camera matrices.
function observeField() {
  const probe = { last: null, pinned: null, lastDraw: null };
  window.__operativeQA = probe;
  const events = window.__THREE_DEVTOOLS__ ??= new EventTarget();
  events.addEventListener('observe', ({ detail: renderer }) => {
    if (!renderer.isWebGLRenderer) return;
    const render = renderer.render.bind(renderer);
    renderer.render = (scene, camera) => {
      const host = renderer.domElement.closest('.operative-field-canvas');
      if (!host || !camera.isOrthographicCamera) return render(scene, camera);
      probe.last = { renderer, scene, camera };
      let drawn = camera;
      if (probe.pinned) {
        drawn = camera.clone();
        drawn.matrixWorld.fromArray(probe.pinned.matrixWorld);
        drawn.matrixWorld.decompose(drawn.position, drawn.quaternion, drawn.scale);
        drawn.matrix.copy(drawn.matrixWorld);
        drawn.matrixAutoUpdate = false;
        drawn.matrixWorldAutoUpdate = false;
        drawn.matrixWorldInverse.copy(drawn.matrixWorld).invert();
        drawn.projectionMatrix.fromArray(probe.pinned.projectionMatrix);
        drawn.projectionMatrixInverse.copy(drawn.projectionMatrix).invert();
      }
      render(scene, drawn);
      const tool = scene.getObjectByName('hollow-aspirator');
      let tipViewport = null;
      if (tool?.visible) {
        const tip = tool.position.clone().set(0, 0, 0);
        tool.localToWorld(tip).project(drawn);
        tipViewport = [(tip.x + 1) / 2, (1 - tip.y) / 2];
      }
      probe.lastDraw = {
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
