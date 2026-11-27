import * as THREE from 'three';
import type {Affine, Point3} from './coordinates';
import {inverseAffine} from './coordinates.ts';
import type {ViewerVolume} from './contracts';
import {paintSection, sampleSection, sectionWindow} from './operativeSection.ts';

const STEP_MM = 0.5, SIZE_MM = 170;

/** The face of an inspection cutaway through a host head. Without it the cut
 * shows the inside walls of every layer's mesh. The cap lies on the cut plane
 * and shows, for each point, the layer owning its engine cell shaded by the
 * case scan intensity there; free space without scan signal stays open. */
export class SectionCap {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>;
  private key = '';
  private volume?: ViewerVolume;
  constructor() {
    const material = new THREE.MeshBasicMaterial({side: THREE.DoubleSide, alphaTest: 0.5, toneMapped: false});
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    this.mesh.name = 'inspection-section-cap';
    this.mesh.visible = false;
    this.mesh.renderOrder = -1;
    this.mesh.frustumCulled = false;
  }
  /** Show the cap on `plane` (kept side: signed distance >= 0) for `volume`; null hides it. */
  update(plane: THREE.Plane | null, volume: ViewerVolume, rasAffine: Affine, centre: THREE.Vector3) {
    if (!plane) {
      this.mesh.visible = false;
      return false;
    }
    const key = [plane.normal.toArray().map((n) => n.toFixed(5)), plane.constant.toFixed(4)].join('|');
    if (key === this.key && volume === this.volume) {
      this.mesh.visible = true;
      return false;
    }
    this.key = key;
    this.volume = volume;
    const normal = plane.normal.clone().normalize();
    const reference = Math.abs(normal.z) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
    const across = new THREE.Vector3().crossVectors(reference, normal).normalize();
    const down = new THREE.Vector3().crossVectors(normal, across).normalize();
    const onPlane = centre.clone().addScaledVector(normal, -plane.distanceToPoint(centre));
    const size = Math.round(SIZE_MM / STEP_MM);
    const origin = onPlane.clone().addScaledVector(across, -SIZE_MM / 2).addScaledVector(down, -SIZE_MM / 2);
    const layers = volume.compartments.filter((layer) => layer.mask.length === volume.mri.length);
    const sample = sampleSection(layers, volume.mri, volume.shape, inverseAffine(rasAffine), {
      origin: origin.toArray() as Point3, across: across.toArray() as Point3, down: down.toArray() as Point3,
      width: size, height: size, stepMm: STEP_MM,
    });
    const window = sectionWindow(sample);
    // Unlabelled space without scan signal (air beyond the skin) stays open.
    const open = new Uint8Array(sample.label.length);
    const threshold = window[0] + 0.08 * (window[1] - window[0]);
    for (let i = 0; i < open.length; i++) open[i] = Number(!sample.label[i] && !(sample.signal[i] > threshold));
    const pixels = paintSection(sample, layers, window, open);
    const texture = new THREE.DataTexture(pixels, size, size, THREE.RGBAFormat);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const corner = (x: number, y: number) =>
      origin.clone().addScaledVector(across, x * SIZE_MM).addScaledVector(down, y * SIZE_MM).toArray();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([...corner(0, 0), ...corner(1, 0), ...corner(1, 1), ...corner(0, 1)], 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.mesh.geometry.dispose();
    this.mesh.geometry = geometry;
    this.mesh.material.map?.dispose();
    this.mesh.material.map = texture;
    this.mesh.material.needsUpdate = true;
    this.mesh.visible = true;
    return true;
  }
  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.map?.dispose();
    this.mesh.material.dispose();
  }
}
