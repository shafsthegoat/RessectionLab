import * as THREE from 'three';
import type {Affine, Point3} from './coordinates';
import {inverseAffine} from './coordinates.ts';
import type {ViewerVolume} from './contracts';
import {paintSection, sampleSection, sectionWindow} from './operativeSection.ts';
import {sampleTrilinear} from './coordinates.ts';
import {CAP_LINE, CAP_REGIONS, capAxes, capRegions, paintCap, planeSegments, type CapRegion, type CapSegment, type CapSurface, type CapSurfaceName} from './sourceSectionCap.ts';

const STEP_MM = 0.5, SIZE_MM = 170, SOURCE_STEP_MM = 0.3;

/** The source-surface scene's description of the cut: its surfaces (RAS), the
 * exterior normal and which tissue the engine has opened or removed. */
export interface SourceCapInput {
  key: string;
  outward: Point3;
  surfaces: CapSurface[];
  /** Surfaces drawn only as boundary lines (the assumed dura). */
  lineSurfaces: Array<Omit<CapSurface, 'name'> & {name: 'dura'}>;
  /** Whether a point of `region` still holds tissue (not in an opened flap footprint or the cavity). */
  closed: (ras: Point3, region: CapRegion) => boolean;
  /** Whether a boundary of `surface` at a point is still present. */
  lineKept: (ras: Point3, surface: CapSurfaceName | 'dura') => boolean;
}

/** The face of an inspection cutaway through a host head. Without it the cut
 * shows the inside walls of every layer's mesh. The cap lies on the cut plane
 * and shows, for each point, the layer owning its engine cell shaded by the
 * case scan intensity there; free space without scan signal stays open. */
export class SectionCap {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>;
  /** Layer boundaries on the cap (source surfaces cut by the plane). */
  readonly lines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  /** Milliseconds of the last cap build. */
  buildMs = 0;
  private key = '';
  private volume?: ViewerVolume;
  constructor() {
    const material = new THREE.MeshBasicMaterial({side: THREE.DoubleSide, alphaTest: 0.5, toneMapped: false,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1});
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    this.mesh.name = 'inspection-section-cap';
    this.mesh.visible = false;
    this.mesh.renderOrder = -1;
    this.mesh.frustumCulled = false;
    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({vertexColors: true, toneMapped: false}));
    this.lines.name = 'inspection-section-lines';
    this.lines.frustumCulled = false;
    this.lines.raycast = () => {};
    this.mesh.add(this.lines);
  }
  /** Show the cap on `plane` (kept side: signed distance >= 0) for `volume`; null hides it. */
  update(plane: THREE.Plane | null, volume: ViewerVolume, rasAffine: Affine, centre: THREE.Vector3, source?: SourceCapInput | null) {
    if (!plane) {
      this.mesh.visible = false;
      return false;
    }
    const key = [plane.normal.toArray().map((n) => n.toFixed(5)), plane.constant.toFixed(4), source?.key ?? ''].join('|');
    if (key === this.key && volume === this.volume) {
      this.mesh.visible = true;
      return false;
    }
    this.key = key;
    this.volume = volume;
    const started = performance.now();
    if (source) {
      this.updateFromSource(plane, volume, rasAffine, centre, source);
      this.buildMs = performance.now() - started;
      return true;
    }
    this.lines.visible = false;
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
    this.buildMs = performance.now() - started;
    return true;
  }
  /** The cap from the source surfaces: smooth layer regions filled with the scan, boundaries as lines. */
  private updateFromSource(plane: THREE.Plane, volume: ViewerVolume, rasAffine: Affine, centre: THREE.Vector3, source: SourceCapInput) {
    const normal = plane.normal.clone().normalize();
    const {across, down} = capAxes(normal.toArray() as Point3, source.outward);
    const a = new THREE.Vector3(...across), d = new THREE.Vector3(...down);
    const onPlane = centre.clone().addScaledVector(normal, -plane.distanceToPoint(centre));
    const size = Math.round(SIZE_MM / SOURCE_STEP_MM), step = SIZE_MM / size;
    const origin = onPlane.clone().addScaledVector(a, -SIZE_MM / 2).addScaledVector(d, -SIZE_MM / 2);
    const o = origin.toArray() as Point3, n = normal.toArray() as Point3;
    const segments = new Map<CapSurfaceName, CapSegment[]>();
    for (const surface of source.surfaces) segments.set(surface.name, planeSegments(surface, o, n, across, down));
    const regions = capRegions(segments, {width: size, height: size, stepMm: step});
    const toVoxel = inverseAffine(rasAffine), [nx, ny, nz] = volume.shape;
    const signal = new Float32Array(regions.length).fill(NaN), closed = new Uint8Array(regions.length);
    const p: Point3 = [0, 0, 0], voxel: Point3 = [0, 0, 0];
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const i = y * size + x;
      if (!regions[i]) continue;
      const u = (x + 0.5) * step, v = (y + 0.5) * step;
      for (let k = 0; k < 3; k++) p[k] = o[k] + across[k] * u + down[k] * v;
      for (let k = 0; k < 3; k++) voxel[k] = toVoxel[k][0] * p[0] + toVoxel[k][1] * p[1] + toVoxel[k][2] * p[2] + toVoxel[k][3];
      if (voxel[0] < -0.5 || voxel[1] < -0.5 || voxel[2] < -0.5 || voxel[0] > nx - 0.5 || voxel[1] > ny - 0.5 || voxel[2] > nz - 0.5) continue;
      signal[i] = sampleTrilinear(volume.mri, volume.shape, voxel);
      closed[i] = Number(source.closed([p[0], p[1], p[2]], CAP_REGIONS[regions[i]]));
    }
    const pixels = paintCap(regions, signal, (i) => closed[i] === 1);
    const texture = new THREE.DataTexture(pixels, size, size, THREE.RGBAFormat);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const corner = (x: number, y: number) => origin.clone().addScaledVector(a, x * SIZE_MM).addScaledVector(d, y * SIZE_MM).toArray();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([...corner(0, 0), ...corner(1, 0), ...corner(1, 1), ...corner(0, 1)], 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.mesh.geometry.dispose();
    this.mesh.geometry = geometry;
    this.mesh.material.map?.dispose();
    this.mesh.material.map = texture;
    this.mesh.material.needsUpdate = true;
    // Boundaries: every surface's segments where its tissue remains, a hair in front of the cap.
    const positions: number[] = [], colours: number[] = [], colour = new THREE.Color();
    const at = (u: number, v: number) => [0, 1, 2].map((k) => o[k] + across[k] * u + down[k] * v - n[k] * 0.03);
    const draw = (name: CapSurfaceName | 'dura', list: CapSegment[]) => {
      colour.set(CAP_LINE[name]);
      for (const s of list) {
        const mid = at((s.u0 + s.u1) / 2, (s.v0 + s.v1) / 2) as Point3;
        if (!source.lineKept(mid, name)) continue;
        positions.push(...at(s.u0, s.v0), ...at(s.u1, s.v1));
        colours.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
      }
    };
    for (const [name, list] of segments) draw(name, list);
    for (const surface of source.lineSurfaces) draw('dura', planeSegments({...surface, name: 'skin'}, o, n, across, down));
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    lineGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
    this.lines.geometry.dispose();
    this.lines.geometry = lineGeometry;
    this.lines.visible = true;
    this.mesh.visible = true;
  }
  dispose() {
    this.lines.geometry.dispose();
    this.lines.material.dispose();
    this.mesh.geometry.dispose();
    this.mesh.material.map?.dispose();
    this.mesh.material.dispose();
  }
}
