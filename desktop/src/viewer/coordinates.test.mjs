import test from "node:test";
import assert from "node:assert/strict";
import {
  cIndex,
  inverseAffine,
  rasAffine,
  sampleTrilinear,
  slicePoint,
  sliceRect,
  transformPoint,
  volumeBounds,
} from "./coordinates.ts";
import { maskSurface } from "./surface.ts";
import { surfaceNormals } from "./surfaceNormals.ts";
import * as THREE from "three";
import { physicalBounds, placeInSourceFrame } from "./sceneGeometry.ts";
import { residualMask, validateReplay } from "./replay.ts";
import {
  COMPARISON_COLORS,
  FAILURE_COLOR,
  routeAppearance,
} from "./routeAppearance.ts";
import { selectComparisonRoutes } from "../route-selection.ts";
import { createHash } from "node:crypto";

const close = (actual, expected) =>
  actual.forEach((value, index) =>
    assert.ok(
      Math.abs(value - expected[index]) < 1e-8,
      `${actual} != ${expected}`,
    ),
  );

test("Display normal averaging preserves every source triangle position exactly", () => {
  const positions = maskSurface(new Uint8Array(125).fill(1), [5, 5, 5]);
  const digest = () =>
    createHash("sha256").update(new Uint8Array(positions.buffer)).digest("hex");
  const before = digest(),
    normals = surfaceNormals(positions);
  assert.equal(digest(), before);
  assert.equal(normals.length, positions.length);
  const shared = new Map();
  for (let i = 0; i < positions.length; i += 3) {
    const id = [...positions.slice(i, i + 3)].join(","),
      normal = [...normals.slice(i, i + 3)];
    assert.ok(Math.abs(Math.hypot(...normal) - 1) < 1e-7);
    if (shared.has(id)) assert.deepEqual(normal, shared.get(id));
    else shared.set(id, normal);
  }
  assert.ok(
    shared.size < positions.length / 3,
    "Fixture has duplicated triangle vertices",
  );
  const original = new THREE.BufferGeometry().setAttribute(
    "position",
    new THREE.BufferAttribute(positions, 3),
  );
  original.computeBoundingBox();
  const bounds = original.boundingBox.clone();
  original.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  original.computeBoundingBox();
  assert.ok(original.boundingBox.equals(bounds));
  assert.equal(original.getAttribute("position").array, positions);
});

test("Only exactly coincident vertices share display normals", () => {
  const positions = new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
  ]);
  const normals = surfaceNormals(positions);
  const expected = Math.SQRT1_2;
  assert.ok(Math.abs(normals[1] - expected) < 1e-7);
  assert.ok(Math.abs(normals[2] - expected) < 1e-7);
  close([...normals.slice(6, 9)], [0, 0, 1]);
  positions[9] = 0.000001;
  const unjoined = surfaceNormals(positions);
  close([...unjoined.slice(0, 3)], [0, 0, 1]);
  assert.throws(
    () => surfaceNormals(new Float32Array(8)),
    /complete triangles/,
  );
  assert.throws(() => surfaceNormals(new Float32Array(9)), /degenerate/);
});

test("Packed source-grid normal keys preserve separated and negative coordinates", () => {
  const first = [0, 0, 0, 1, 0, 0, 0, 1, 0];
  const second = [-4, -2, -1, -4, -2, 0, -3, -2, -1];
  const positions = new Float32Array([...first, ...second]);
  const normals = surfaceNormals(positions);
  for (let i = 0; i < 9; i += 3) close([...normals.slice(i, i + 3)], [0, 0, 1]);
  for (let i = 9; i < 18; i += 3)
    close([...normals.slice(i, i + 3)], [0, 1, 0]);
  const translated = new Float32Array(positions.map((value) => value + 100000));
  assert.deepEqual(surfaceNormals(translated), normals);
});

test("Clearing route A retains B's amber comparison identity in the viewer", () => {
  const routes = [
    { route_id: "one" },
    { route_id: "two", category: "rejected" },
  ];
  const both = selectComparisonRoutes(routes, "one", "two");
  const onlyB = selectComparisonRoutes(routes, "", "two");
  assert.deepEqual(routeAppearance(both[1], 1), {
    slot: "B",
    color: COMPARISON_COLORS.B,
  });
  assert.deepEqual(routeAppearance(onlyB[0], 0), routeAppearance(both[1], 1));
  assert.equal(routeAppearance(onlyB[0], 0).color, "#e5c598");
  assert.notEqual(
    routeAppearance(onlyB[0], 0).color,
    FAILURE_COLOR,
    "Failure markers have separate semantics from route identity",
  );
  assert.deepEqual(routeAppearance({}, 0), {
    slot: "A",
    color: COMPARISON_COLORS.A,
  });
  assert.deepEqual(routeAppearance({}, 1), {
    slot: "B",
    color: COMPARISON_COLORS.B,
  });
});

function replayFixture() {
  const volume = {
    caseId: "overlapping-source-test",
    caseHash: "case-version-a",
    frame: "RAS+",
    affine: [
      [0, -3, 0, 10],
      [2, 0, 0, -20],
      [0, 0, 4, 7],
      [0, 0, 0, 1],
    ],
    shape: [2, 2, 2],
    mri: new Float32Array(8),
    compartments: [
      {
        name: "A",
        color: "#ee9988",
        mask: new Uint8Array([1, 1, 0, 0, 0, 0, 0, 0]),
      },
      {
        name: "B",
        color: "#ddcc77",
        mask: new Uint8Array([0, 1, 1, 0, 0, 0, 0, 0]),
      },
    ],
  };
  const replay = {
    caseHash: volume.caseHash,
    scope: "native-source-grid",
    independentlyAccepted: true,
    shape: [2, 2, 2],
    affine: volume.affine.map((row) => [...row]),
    step: 2,
    stepCount: 3,
    removedMask: new Uint8Array([0, 1, 0, 0, 0, 0, 0, 1]),
    removedTargetVolumeMm3: 24,
    removedNormalVolumeMm3: 24,
    residualTargetVolumeMm3: 48,
  };
  return { volume, replay };
}

test("Accepted replay accounts for unique target cells and preserves source arrays", () => {
  const { volume, replay } = replayFixture();
  validateReplay(volume, replay);
  const source = volume.compartments[0].mask.slice(),
    effect = replay.removedMask.slice();
  assert.deepEqual(
    residualMask(source, effect),
    new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]),
  );
  assert.deepEqual(source, volume.compartments[0].mask);
  assert.deepEqual(effect, replay.removedMask);
  assert.throws(
    () => residualMask(source, new Uint8Array(1)),
    /identical source grids/,
  );
});

test("Replay gate rejects unaccepted, stale, misframed, nonbinary and inconsistent effects", () => {
  const invalid = [
    { independentlyAccepted: false },
    { scope: "coarse-grid" },
    { caseHash: "stale-case" },
    { step: -1 },
    { step: 4 },
    { step: 1.5 },
    { stepCount: -1 },
    { shape: [2, 2, 3] },
    { removedMask: new Uint8Array(7) },
    { removedMask: new Uint8Array([0, 2, 0, 0, 0, 0, 0, 1]) },
    { removedMask: new Float32Array(8) },
    { removedTargetVolumeMm3: 48 },
    { removedNormalVolumeMm3: 0 },
    { residualTargetVolumeMm3: 24 },
    { removedTargetVolumeMm3: NaN },
    { removedNormalVolumeMm3: -1 },
    {
      affine: [
        [0, -3, 0, 11],
        [2, 0, 0, -20],
        [0, 0, 4, 7],
        [0, 0, 0, 1],
      ],
    },
  ];
  for (const patch of invalid) {
    const { volume, replay } = replayFixture();
    assert.throws(
      () => validateReplay(volume, { ...replay, ...patch }),
      undefined,
      JSON.stringify(patch),
    );
  }
  const { volume, replay } = replayFixture();
  volume.affine[2] = [0, 0, 0, 7];
  replay.affine = volume.affine;
  assert.throws(() => validateReplay(volume, replay), /Source geometry/);
});

test("Replay affine conversion preserves native LPS voxel indexing in RAS", () => {
  const { volume, replay } = replayFixture();
  volume.frame = "LPS+";
  assert.throws(() => validateReplay(volume, replay), /affine differs/);
  replay.affine = rasAffine(volume.affine, volume.frame);
  validateReplay(volume, replay);
  const sourceCell = [1, 0, 0];
  close(transformPoint(replay.affine, sourceCell), [-10, 18, 7]);
});

test("Zero-removal STOP replay retains every source target cell", () => {
  const { volume, replay } = replayFixture();
  replay.removedMask.fill(0);
  replay.step = 0;
  replay.stepCount = 0;
  replay.removedTargetVolumeMm3 = 0;
  replay.removedNormalVolumeMm3 = 0;
  replay.residualTargetVolumeMm3 = 72;
  validateReplay(volume, replay);
  for (const layer of volume.compartments)
    assert.deepEqual(residualMask(layer.mask, replay.removedMask), layer.mask);
});

test("C-order scalar lookup and trilinear interpolation preserve z-fastest source data", () => {
  const shape = [2, 3, 4],
    data = new Float32Array(24);
  for (let x = 0; x < 2; x++)
    for (let y = 0; y < 3; y++)
      for (let z = 0; z < 4; z++)
        data[cIndex(shape, [x, y, z])] = 100 * x + 10 * y + z;
  assert.equal(sampleTrilinear(data, shape, [0.5, 0.5, 1.5]), 56.5);
  assert.equal(sampleTrilinear(data, shape, [1, 2, 3]), 123);
  assert.equal(sampleTrilinear(data, shape, [-0.1, 2, 3]), 0);
});

test("Oblique anisotropic source affine is invertible in physical millimetres", () => {
  const angle = Math.PI / 6,
    c = Math.cos(angle),
    s = Math.sin(angle);
  const affine = [
    [2 * c, -3 * s, 0, 10],
    [2 * s, 3 * c, 0, -20],
    [0, 0, 4, 7],
    [0, 0, 0, 1],
  ];
  const voxel = [3.2, 11, 0.7],
    world = transformPoint(affine, voxel);
  close(transformPoint(inverseAffine(affine), world), voxel);
  const bounds = volumeBounds([10, 20, 5], affine);
  assert.ok(bounds[0][0] < 10);
  assert.equal(bounds[0][2], 7);
  assert.equal(bounds[1][2], 23);
  assert.throws(
    () =>
      inverseAffine([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ]),
    /singular/,
  );
});

test("LPS source is transformed once into RAS display coordinates", () => {
  const affine = rasAffine(
    [
      [2, 0, 0, 10],
      [0, 3, 0, 20],
      [0, 0, 4, 30],
      [0, 0, 0, 1],
    ],
    "LPS+",
  );
  close(transformPoint(affine, [1, 2, 3]), [-12, -26, 42]);
  close(transformPoint(inverseAffine(affine), [-12, -26, 42]), [1, 2, 3]);
  assert.throws(() => rasAffine(affine, "unknown"), /Unsupported/);
});

test("Linked MPR click mapping uses neurological orientation and equal physical scale", () => {
  const bounds = [
      [-20, -30, -40],
      [80, 170, 60],
    ],
    cursor = [10, 15, 25],
    rect = sliceRect(bounds, "axial", 320, 180);
  assert.equal(rect.width / 100, rect.height / 200);
  close(
    slicePoint(bounds, "axial", cursor, rect, rect.left, rect.top),
    [-20, 170, 25],
  );
  close(
    slicePoint(
      bounds,
      "axial",
      cursor,
      rect,
      rect.left + rect.width,
      rect.top + rect.height,
    ),
    [80, -30, 25],
  );
  assert.equal(slicePoint(bounds, "axial", cursor, rect, 0, 0), null);
  const sagittal = sliceRect(bounds, "sagittal", 320, 180);
  close(
    slicePoint(
      bounds,
      "sagittal",
      cursor,
      sagittal,
      sagittal.left,
      sagittal.top,
    ),
    [10, -30, 60],
  );
});

test("Surface worker derives only source-mask geometry and pads image-edge labels", () => {
  const mask = new Uint8Array(27);
  mask[13] = 1;
  const vertices = maskSurface(mask, [3, 3, 3]);
  assert.ok(vertices.length > 0);
  assert.equal(vertices.length % 9, 0);
  for (let axis = 0; axis < 3; axis++) {
    const coords = Array.from(vertices).filter((_, i) => i % 3 === axis);
    assert.equal(Math.min(...coords), 0.5);
    assert.equal(Math.max(...coords), 1.5);
  }
  const boundary = new Uint8Array(8);
  boundary[0] = 1;
  assert.equal(Math.min(...maskSurface(boundary, [2, 2, 2])), -0.5);
  assert.equal(maskSurface(new Uint8Array(8), [2, 2, 2]).length, 0);
});

test("A newly added surface has physical camera bounds before its first render", () => {
  const group = new THREE.Group(),
    affine = [
      [-1, 0, 0, 0],
      [0, -1, 0, 239],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
  const mesh = () =>
    new THREE.Mesh(
      new THREE.BufferGeometry().setAttribute(
        "position",
        new THREE.Float32BufferAttribute(
          [150, 140, 90, 180, 170, 110, 170, 160, 100],
          3,
        ),
      ),
      new THREE.MeshBasicMaterial(),
    );
  const first = mesh();
  placeInSourceFrame(first, affine);
  group.add(first);
  group.updateMatrixWorld(true);
  const newest = mesh();
  placeInSourceFrame(newest, affine);
  group.add(newest);
  const bounds = physicalBounds(group);
  close(bounds.min.toArray(), [-180, 69, 90]);
  close(bounds.max.toArray(), [-150, 99, 110]);
  assert.ok(
    bounds.max.x < 0,
    "No positive native voxel coordinates may leak into the RAS bounding box",
  );
});

test("The source isosurface is watertight, oriented, and has nonzero physical volume", () => {
  const vertices = maskSurface(new Uint8Array(125).fill(1), [5, 5, 5]);
  const edges = new Map();
  let signedVolume = 0;
  for (let index = 0; index < vertices.length; index += 9) {
    const a = [...vertices.slice(index, index + 3)];
    const b = [...vertices.slice(index + 3, index + 6)];
    const c = [...vertices.slice(index + 6, index + 9)];
    signedVolume +=
      (a[0] * (b[1] * c[2] - b[2] * c[1]) +
        a[1] * (b[2] * c[0] - b[0] * c[2]) +
        a[2] * (b[0] * c[1] - b[1] * c[0])) /
      6;
    const ab = b.map((v, i) => v - a[i]),
      ac = c.map((v, i) => v - a[i]);
    assert.ok(
      Math.hypot(
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
      ) > 0,
    );
    for (const [start, end] of [
      [a, b],
      [b, c],
      [c, a],
    ]) {
      const key = [start.join(","), end.join(",")].sort().join("|");
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  assert.ok(
    [...edges.values()].every((count) => count === 2),
    "Every surface edge must have exactly two incident faces",
  );
  // The binary 0.5 isosurface rounds block edges: n³ - 3n/4 + 1/4.
  // Quantitative target volume continues to use the 125 source voxel cells.
  assert.ok(Math.abs(signedVolume - 121.5) < 1e-8);
  assert.ok(
    signedVolume > 0,
    "Outward winding must give positive signed volume",
  );
  const affine = [
    [2, 0, 0, 10],
    [0, 3, 0, -20],
    [0, 0, 4, 7],
    [0, 0, 0, 1],
  ];
  const transformed = [];
  for (let i = 0; i < vertices.length; i += 3)
    transformed.push(...transformPoint(affine, [...vertices.slice(i, i + 3)]));
  let physicalVolume = 0;
  for (let i = 0; i < transformed.length; i += 9) {
    const a = transformed.slice(i, i + 3),
      b = transformed.slice(i + 3, i + 6),
      c = transformed.slice(i + 6, i + 9);
    physicalVolume +=
      (a[0] * (b[1] * c[2] - b[2] * c[1]) +
        a[1] * (b[2] * c[0] - b[0] * c[2]) +
        a[2] * (b[0] * c[1] - b[1] * c[0])) /
      6;
  }
  assert.ok(Math.abs(physicalVolume - 121.5 * 24) < 1e-7);
});
