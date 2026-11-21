import * as THREE from 'three';
import type {ExposureBodyName, OperativeExposureBody, OperativeExposureObject} from '../operative-exposure-types';
import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import {bodyNames, bodyIsDisplaced, sourceCenter} from './exposureAnatomy.ts';
import {buildExposureSurface, ExposureBodyTransport} from './exposureBodySurface.ts';
import {exteriorSkinSurface} from './exteriorSkinSurface.ts';
import {placeInSourceFrame} from './sceneGeometry.ts';
import {generatedTissueFinish} from './generatedTissueFinish.ts';
import {applyOperativeSurface} from './operativeMaterial.ts';
import {layerAppearance, type ViewerWorkspaceSettings} from './workspaceSettings.ts';

type BodyMesh = ReturnType<typeof buildExposureSurface> & {
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
};
type BodyDisplay = {
  members: string;
  transport: ExposureBodyTransport;
  geometryKey?: string;
  seamKey?: string;
  seam?: THREE.Mesh;
  surfaces: BodyMesh[];
};
const up = new THREE.Vector3(0, 0, 1);
function discard(group: THREE.Object3D) {
  group.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.geometry.dispose();
      o.material.dispose();
    }
  });
  group.removeFromParent();
}
function solid(geometry: THREE.BufferGeometry, color: string, metal = 0.0) {
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({color, metalness: metal, roughness: metal ? 0.38 : 0.72, side: THREE.DoubleSide}),
  );
}
function bar(group: THREE.Group, a: THREE.Vector3, b: THREE.Vector3, r: number, color: string, metal = 0) {
  const dir = b.clone().sub(a);
  const mesh = solid(new THREE.CylinderGeometry(r, r, dir.length(), 8), color, metal);
  mesh.position.copy(a).lerp(b, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  group.add(mesh);
}
function fixture(o: OperativeExposureObject) {
  const group = new THREE.Group();
  group.name = 'exposure-object:' + o.id;
  const r = o.radiusMm;
  const span = o.endpointsMm
    ? new THREE.Vector3(...o.endpointsMm[0]).distanceTo(new THREE.Vector3(...o.endpointsMm[1]))
    : undefined;
  const pt = (x: number, y: number, z: number) => new THREE.Vector3(x * r, y * r, z * r);
  if (o.kind === 'raney-clip' || o.kind === 'retaining-stay') {
    const color = o.kind === 'raney-clip' ? '#a6a3a0' : '#b8a38b';
    for (const side of [-1, 1]) {
      bar(group, pt(-0.58, side * 0.26, 0.48), pt(0.6, side * 0.34, 0.3), 0.09 * r, color, 0.6);
      bar(group, pt(0.6, side * 0.34, 0.3), pt(0.66, side * 0.18, -0.16), 0.09 * r, color, 0.6);
    }
    bar(group, pt(-0.58, -0.26, 0.48), pt(-0.58, 0.26, 0.48), 0.13 * r, color, 0.6);
  } else if (o.kind === 'suture') {
    const length = span ?? 1.6 * r;
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0.22 * length, 0.11 * r, 0.4 * r),
      new THREE.Vector3(0.66 * length, -0.05 * r, 0.43 * r),
      new THREE.Vector3(length, 0, 0),
    ]);
    group.add(solid(new THREE.TubeGeometry(curve, 12, 0.1 * r, 6, false), '#665963'));
  } else if (o.kind === 'bone-anchor') {
    const length = span ?? 1.2 * r;
    const plate = solid(new THREE.BoxGeometry(length + 0.5 * r, 0.62 * r, 0.17 * r), '#a3a9aa', 0.67);
    plate.position.set(length * 0.5, 0, 0.1 * r);
    group.add(plate);
    for (const x of [0, length]) {
      const head = solid(new THREE.CylinderGeometry(0.19 * r, 0.22 * r, 0.11 * r, 8), '#7b8183', 0.68);
      head.rotation.x = Math.PI / 2;
      head.position.set(x, 0, 0.22 * r);
      group.add(head);
    }
  } else {
    const pad = solid(new THREE.SphereGeometry(r, 14, 8), o.kind === 'cottonoid' ? '#dddbca' : '#a57f6a');
    pad.scale.set(0.94, 0.68, 0.42);
    group.add(pad);
    if (o.kind === 'cottonoid') bar(group, pt(-0.7, 0, 0.31), pt(0.7, 0, 0.31), 0.065 * r, '#748c9a');
  }
  group.userData.operativeObject = {id: o.id, kind: o.kind, radiusMm: r};
  return group;
}

/** Only accepted severed edges are drawn. The two endpoints lie on opposite
 * sides of a real neighbor face. Position a narrow mark at its outer lip;
 * intervening skin, bone and fixtures retain their ordinary depth occlusion. */
function seams(
  name: ExposureBodyName,
  body: OperativeExposureBody,
  transport: ExposureBodyTransport,
  view: SurgicalView,
  outward: THREE.Vector3,
) {
  const coordinates: number[] = [];
  const spacing = Math.min(...view.scene.resolutionMm);
  const width = (name === 'scalp' ? 0.13 : 0.1) * spacing;
  const key = (p: THREE.Vector3) =>
    p
      .toArray()
      .map((v) => Math.round(1000 * v))
      .join(',');
  type Face = {anchor: number; center: THREE.Vector3; ends: THREE.Vector3[]; towardFixed: THREE.Vector3};
  const faces: Face[] = [],
    meeting = new Map<string, Face[]>();
  for (const [moving, fixed] of body.seamEdges) {
    const anchor = transport.byId.get(moving);
    if (anchor === undefined) continue;
    const p = transport.rest[anchor],
      q = new THREE.Vector3(...sourceCenter(fixed, view.scene.shape, view.scene.affine));
    const outwardStep = q.clone().sub(p),
      side = new THREE.Vector3().crossVectors(outward, outwardStep);
    if (side.lengthSq() < 1e-8) continue;
    side.normalize().multiplyScalar(spacing * 0.5);
    const center = p
      .clone()
      .lerp(q, 0.5)
      .addScaledVector(outward, 0.527 * spacing);
    const face = {
      anchor,
      center,
      ends: [center.clone().sub(side), center.clone().add(side)],
      towardFixed: outwardStep.normalize(),
    };
    faces.push(face);
    for (const end of face.ends) {
      const k = key(end),
        group = meeting.get(k) ?? [];
      group.push(face);
      meeting.set(k, group);
    }
  }
  const tri = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3) =>
    coordinates.push(...a.toArray(), ...b.toArray(), ...c.toArray());
  for (const face of faces) {
    // Round only between physically severed neighboring faces. A partial
    // incision ends at its actual face boundary; disjoint cuts stay disjoint.
    const endpoints = face.ends.map((end) => {
      const others = meeting.get(key(end))!.filter((other) => other !== face);
      return others.length === 1 ? face.center.clone().lerp(others[0].center, 0.5) : end;
    });
    const line = new THREE.QuadraticBezierCurve3(endpoints[0], face.center, endpoints[1]);
    let previous: THREE.Vector3[] | undefined;
    for (let part = 0; part <= 5; part++) {
      const t = part / 5,
        center = line.getPoint(t),
        tangent = line.getTangent(t);
      const across = new THREE.Vector3().crossVectors(tangent, outward).normalize().multiplyScalar(width);
      if (across.dot(face.towardFixed) < 0) across.negate();
      const row = [
        center.clone().add(across),
        center,
        transport.point(center, face.anchor),
        transport.point(center.clone().sub(across), face.anchor),
      ];
      if (previous)
        for (const side of [0, 2]) {
          tri(previous[side], row[side], previous[side + 1]);
          tri(previous[side + 1], row[side], row[side + 1]);
        }
      previous = row;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(coordinates, 3));
  geo.computeVertexNormals();
  const mesh = solid(geo, name === 'scalp' ? '#924f4a' : name === 'dura' ? '#8c635e' : '#816752');
  mesh.name = 'accepted-seams:' + name;
  return mesh;
}
export class OperativeExposureScene {
  readonly group = new THREE.Group();
  private rest?: THREE.Group;
  private restKey?: string;
  private smooth = true;
  private exterior?: {
    mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
    support: Set<number>;
    unchanged: boolean;
    opaque: boolean;
  };
  private bodies = new Map<ExposureBodyName, BodyDisplay>();
  private markers = new THREE.Group();
  private fixtures = new Map<string, {model: THREE.Group; key: string}>();
  private outward = up.clone();
  private source?: Uint16Array;
  private view?: SurgicalView;
  private volume?: ViewerVolume;
  constructor() {
    this.group.name = 'operative-exposure';
    this.group.add(this.markers);
  }
  get picks() {
    return [...this.bodies.values()].flatMap((b) => b.surfaces.map((s) => s.mesh).filter((m) => m.visible));
  }
  private refreshExterior() {
    if (this.exterior) this.exterior.mesh.visible = this.smooth && this.exterior.unchanged && this.exterior.opaque;
  }
  setSmooth(enabled: boolean) {
    this.smooth = enabled;
    this.refreshExterior();
    for (const display of this.bodies.values())
      for (const s of display.surfaces) s.mesh.geometry = enabled ? s.geometry : s.sourceGeometry();
  }
  useSourceForPick(displayed: Map<THREE.Mesh, THREE.BufferGeometry>) {
    for (const display of this.bodies.values())
      for (const s of display.surfaces)
        if (s.mesh.visible) {
          displayed.set(s.mesh, s.mesh.geometry);
          s.mesh.geometry = s.sourceGeometry();
        }
  }
  clear() {
    for (const display of this.bodies.values()) for (const s of display.surfaces) s.disposeOther(s.mesh.geometry);
    for (const child of [...this.group.children]) discard(child);
    this.bodies.clear();
    this.fixtures.clear();
    this.exterior = undefined;
    this.rest = undefined;
    this.restKey = undefined;
    this.markers = new THREE.Group();
    this.group.add(this.markers);
    this.source = undefined;
    this.view = undefined;
  }
  update(view: SurgicalView, volume: ViewerVolume, owner: Uint16Array) {
    const state = view.operativeExposureV1;
    if (!state) return;
    if (this.source !== owner) {
      this.clear();
      const layerOf = (kind: string) => volume.compartments.find((layer) => (layer.displayKind ?? layer.kind) === kind);
      const skin = layerOf('scalp'),
        galea = layerOf('galea');
      const skinId = view.scene.regions.find((r) => r.name === skin?.name)?.id;
      const galeaId = view.scene.regions.find((r) => r.name === galea?.name)?.id;
      if (skin && skinId && galeaId) {
        const contour = exteriorSkinSurface(
          owner,
          view.scene.shape,
          view.scene.affine,
          skinId,
          galeaId,
          new Set(state.bodies.scalp.cells),
        );
        if (contour) {
          const {surface} = contour,
            geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.BufferAttribute(surface.smoothPositions!, 3));
          geometry.setAttribute('normal', new THREE.BufferAttribute(surface.smoothNormals!, 3));
          geometry.setIndex(new THREE.BufferAttribute(surface.indices, 1));
          const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({side: THREE.DoubleSide}));
          generatedTissueFinish(mesh.material, 'scalp', false);
          mesh.userData.compartment = skin.name;
          mesh.name = 'intact-exterior-skin-faces';
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          placeInSourceFrame(mesh, view.scene.affine);
          this.group.add(mesh);
          this.exterior = {mesh, support: contour.sourceSupport, unchanged: true, opaque: false};
        }
      }
      const center = (b: OperativeExposureBody) =>
        b.cells
          .reduce(
            (v, id) => v.add(new THREE.Vector3(...sourceCenter(id, view.scene.shape, view.scene.affine))),
            new THREE.Vector3(),
          )
          .divideScalar(b.cells.length);
      this.outward = state.exteriorNormalRas
        ? new THREE.Vector3(...state.exteriorNormalRas).normalize()
        : center(state.bodies.scalp).sub(center(state.bodies.dura)).normalize();
    }
    if (this.view === view && this.volume === volume) return;
    this.source = owner;
    this.view = view;
    this.volume = volume;
    const restKey = state.boneRest ? JSON.stringify(state.boneRest.pads) : undefined;
    if (restKey !== this.restKey) {
      if (this.rest) discard(this.rest);
      this.rest = undefined;
      this.restKey = restKey;
      if (state.boneRest) {
        this.rest = new THREE.Group();
        this.rest.name = 'exposure:bone-rest';
        this.group.add(this.rest);
        const steel = new THREE.Color('#929da0'),
          seat = new THREE.Color('#537775');
        for (const [index, pad] of state.boneRest.pads.entries()) {
          const a = new THREE.Vector3(...pad.aMm),
            b = new THREE.Vector3(...pad.bMm),
            axis = b.clone().sub(a);
          const length = axis.length(),
            r = pad.radiusMm;
          const geometry = new THREE.CapsuleGeometry(r, length, 8, 20);
          const position = geometry.attributes.position,
            colors = new Float32Array(position.count * 3);
          for (let i = 0; i < position.count; i++)
            (position.getY(i) >= length * 0.5 - 0.01 * r ? seat : steel).toArray(colors, 3 * i);
          geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          const mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshStandardMaterial({vertexColors: true, roughness: 0.55, metalness: 0.24}),
          );
          mesh.name = 'bone-rest-pad:' + index;
          mesh.position.copy(a).lerp(b, 0.5);
          if (length > 0) mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.divideScalar(length));
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          this.rest.add(mesh);
        }
      }
    }
    if (this.rest) this.rest.userData.engaged = state.boneRest?.engaged;
    if (this.exterior) {
      const {support} = this.exterior;
      this.exterior.unchanged =
        !view.removed.some((id) => support.has(id)) &&
        !view.moved.some(([a, b]) => support.has(a) || support.has(b)) &&
        !state.bodies.scalp.seamEdges.some(([a, b]) => support.has(a) || support.has(b));
      this.refreshExterior();
    }
    const removed = new Set(view.removed);

    for (const name of bodyNames) {
      const body = state.bodies[name],
        members = body.cells.join(',');
      let display = this.bodies.get(name);
      if (!display || display.members !== members) {
        display?.surfaces.forEach((s) => {
          s.disposeOther(s.mesh.geometry);
          discard(s.mesh);
        });
        if (display?.seam) discard(display.seam);
        display = {members, transport: new ExposureBodyTransport(body, view.scene), surfaces: []};
        this.bodies.set(name, display);
      }
      const changed = display.transport.update(body.centersMm);
      const active = bodyIsDisplaced(body, view.scene.shape, view.scene.affine);
      const ids = active ? body.cells.filter((id) => !removed.has(id)) : [];
      const key = ids.join(',');
      if (key !== display.geometryKey) {
        display.surfaces.forEach((s) => {
          s.disposeOther(s.mesh.geometry);
          discard(s.mesh);
        });
        display.surfaces = [];
        for (const region of view.scene.regions) {
          const subset = ids.filter((id) => owner[id] === region.id);
          if (!subset.length) continue;
          const layer = volume.compartments.find((l) => l.name === region.name);
          const surface = buildExposureSurface(
            subset,
            owner,
            display.transport,
            view.scene,
            region.id,
            layer?.displayKind === 'scalp' ? region.id : undefined,
          );
          const material = new THREE.MeshStandardMaterial({side: THREE.DoubleSide});
          if (layer?.generatedFinish) generatedTissueFinish(material, layer.displayKind ?? layer.kind ?? '', true);
          const mesh = new THREE.Mesh(this.smooth ? surface.geometry : surface.sourceGeometry(), material);
          mesh.name = 'exposure:' + name + ':' + region.name;
          mesh.userData.compartment = region.name;
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          this.group.add(mesh);
          display.surfaces.push({...surface, mesh});
        }
        display.geometryKey = key;
      } else if (changed) display.surfaces.forEach((s) => s.update());
      const seamKey = JSON.stringify(body.seamEdges);
      if (changed || display.seamKey !== seamKey) {
        if (display.seam) discard(display.seam);
        display.seam = body.seamEdges.length ? seams(name, body, display.transport, view, this.outward) : undefined;
        if (display.seam) this.markers.add(display.seam);
        display.seamKey = seamKey;
      }
    }
    const current = new Set<string>();
    for (const obj of state.objects) {
      if (obj.status === 'retrieved') continue;
      current.add(obj.id);
      const key = JSON.stringify([obj.kind, obj.radiusMm, obj.endpointsMm]);
      let record = this.fixtures.get(obj.id);
      if (!record || record.key !== key) {
        if (record) discard(record.model);
        record = {model: fixture(obj), key};
        this.fixtures.set(obj.id, record);
        this.markers.add(record.model);
      }
      const model = record.model;
      model.position.fromArray(obj.positionMm);
      const normal = obj.normal ? new THREE.Vector3(...obj.normal).normalize() : this.outward.clone();
      if (!obj.normal && obj.body !== 'field' && obj.sourceCell !== undefined && obj.status !== 'carried') {
        const transport = this.bodies.get(obj.body)!.transport,
          anchor = transport.byId.get(obj.sourceCell);
        if (anchor !== undefined) normal.applyQuaternion(transport.rotation[anchor]);
      }
      if (obj.endpointsMm) {
        const [a, b] = obj.endpointsMm.map((p) => new THREE.Vector3(...p));
        const direction = b.clone().sub(a).normalize(),
          across = new THREE.Vector3().crossVectors(normal, direction).normalize();
        model.position.copy(a);
        model.quaternion.setFromRotationMatrix(
          new THREE.Matrix4().makeBasis(direction, across, new THREE.Vector3().crossVectors(direction, across)),
        );
      } else model.quaternion.setFromUnitVectors(up, normal);
    }
    for (const [id, record] of this.fixtures)
      if (!current.has(id)) {
        discard(record.model);
        this.fixtures.delete(id);
      }
  }
  appearance(volume: ViewerVolume, settings: ViewerWorkspaceSettings, isVisible: (name: string) => boolean) {
    for (const mesh of [...this.picksIncludingHidden(), ...(this.exterior ? [this.exterior.mesh] : [])]) {
      const layer = volume.compartments.find((l) => l.name === mesh.userData.compartment);
      if (!layer) continue;
      const kind = layer.displayKind ?? layer.kind,
        style = layerAppearance(layer.name, layer.color, settings, kind);
      const m = mesh.material;
      m.color.set(style.color);
      m.opacity = style.opacity;
      const transparent = style.opacity < 1;
      if (m.transparent !== transparent) {
        m.transparent = transparent;
        m.needsUpdate = true;
      }
      m.depthWrite = !transparent;
      mesh.visible = isVisible(layer.name);
      applyOperativeSurface(m, kind, settings);
      if (this.exterior?.mesh === mesh) {
        this.exterior.opaque =
          mesh.visible &&
          !transparent &&
          settings.microscope?.tissueStyle !== 'labels' &&
          !settings.microscope?.inspectionCutaway &&
          !settings.microscope?.inspectionTransparency;
        this.refreshExterior();
      }
    }
  }
  private picksIncludingHidden() {
    return [...this.bodies.values()].flatMap((b) => b.surfaces.map((s) => s.mesh));
  }
}
