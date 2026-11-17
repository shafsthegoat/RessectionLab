import * as THREE from 'three';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

// Unitless sculpt in patient coordinates, with the neck pivot at the origin.
// +X is right, +Y is anterior, and +Z is superior.
type Ring = [z: number, halfWidth: number, halfDepth: number, x: number, y: number];

const torso: Ring[] = [
  [-.805, .025, .045, 0, .001], [-.755, .115, .083, 0, .002],
  [-.662, .151, .096, 0, -.007], [-.548, .133, .082, 0, -.004],
  [-.423, .141, .089, 0, -.011], [-.307, .166, .102, 0, -.006],
  [-.202, .184, .099, 0, -.004], [-.122, .168, .081, 0, -.008],
  [-.071, .104, .064, 0, -.01], [-.036, .063, .054, 0, -.011],
  [.025, .059, .05, 0, -.011],
];
const arm: Ring[] = [
  [-.837, .01, .01, .297, .018], [-.805, .026, .018, .296, .013],
  [-.744, .03, .023, .29, .013], [-.703, .025, .022, .284, .01],
  [-.627, .036, .031, .273, .007], [-.522, .041, .04, .252, -.006],
  [-.435, .038, .038, .236, -.014], [-.323, .049, .047, .214, -.009],
  [-.213, .052, .056, .193, -.004], [-.144, .032, .036, .181, -.007],
];
const leg: Ring[] = [
  [-1.507, .028, .057, .11, .055], [-1.481, .042, .098, .11, .057],
  [-1.448, .039, .088, .11, .044], [-1.402, .026, .039, .11, .002],
  [-1.338, .029, .037, .11, -.005], [-1.236, .049, .052, .106, -.016],
  [-1.135, .043, .05, .102, -.001], [-1.053, .044, .053, .097, .01],
  [-.942, .059, .068, .09, .002], [-.818, .072, .083, .08, -.001],
  [-.708, .079, .082, .072, -.003], [-.633, .044, .047, .083, -.007],
];
const headProfile: Ring[] = [
  [-.036, .057, .047, 0, -.009], [.023, .058, .051, 0, -.009],
  [.066, .057, .057, 0, -.006], [.089, .051, .065, 0, .006],
  [.113, .069, .076, 0, .008], [.151, .087, .089, 0, .004],
  [.198, .098, .097, 0, -.001], [.243, .101, .103, 0, -.007],
  [.285, .091, .098, 0, -.01], [.318, .072, .078, 0, -.01],
  [.341, .039, .046, 0, -.009], [.351, .001, .001, 0, -.008],
];

function profileAt(profile: Ring[], z: number): Ring {
  let upper = 1;
  while (upper < profile.length - 1 && z > profile[upper][0]) upper++;
  const a = profile[upper - 1], b = profile[upper];
  const before = profile[Math.max(0, upper - 2)], after = profile[Math.min(profile.length - 1, upper + 1)];
  const t = THREE.MathUtils.clamp((z - a[0]) / (b[0] - a[0]), 0, 1);
  const sample = (axis: number) => {
    const start = a[axis], end = b[axis];
    const startSlope = (b[axis] - before[axis]) * (b[0] - a[0]) / (b[0] - before[0]);
    const endSlope = (after[axis] - a[axis]) * (b[0] - a[0]) / (after[0] - a[0]);
    const t2 = t * t, t3 = t2 * t;
    return start + (end - start) * (3 * t2 - 2 * t3) + startSlope * (t3 - 2 * t2 + t) + endSlope * (t3 - t2);
  };
  return [z, sample(1), sample(2), sample(3), sample(4)];
}

function profileDistance(profile: Ring[], ring: Ring, x: number, y: number): number {
  const [z, width, depth, centreX, centreY] = ring;
  const side = (Math.hypot((x - centreX) / width, (y - centreY) / depth) - 1) * Math.min(width, depth);
  return Math.max(side, profile[0][0] - z, z - profile[profile.length - 1][0]);
}

function blend(a: number, b: number, radius: number): number {
  const overlap = Math.max(radius - Math.abs(a - b), 0) / radius;
  return Math.min(a, b) - overlap * overlap * radius * .25;
}

let bodyGeometry: THREE.BufferGeometry | undefined;
function sculptBody(): THREE.BufferGeometry {
  if (bodyGeometry) return bodyGeometry.clone();

  // Blend the shoulders and hips before extracting the surface so the figure
  // has the continuous finish of a molded mannequin at any viewing angle.
  const resolution = 50, maxTriangles = 18000;
  const material = new THREE.MeshBasicMaterial();
  const surface = new MarchingCubes(resolution, material, false, false, maxTriangles);
  surface.isolation = 0;
  const halfWidth = .4, halfDepth = .21, halfHeight = .85, centreZ = -.73;
  const axis = (index: number) => index * 2 / resolution - 1;
  for (let iz = 0; iz < resolution; iz++) {
    const z = axis(iz) * halfHeight + centreZ;
    const torsoRing = profileAt(torso, z), armRing = profileAt(arm, z), legRing = profileAt(leg, z);
    for (let iy = 0; iy < resolution; iy++) {
      const y = axis(iy) * halfDepth;
      for (let ix = 0; ix < resolution; ix++) {
        const x = Math.abs(axis(ix) * halfWidth);
        const upper = blend(profileDistance(torso, torsoRing, x, y), profileDistance(arm, armRing, x, y), .066);
        const distance = blend(upper, profileDistance(leg, legRing, x, y), .055);
        surface.field[(iz * resolution + iy) * resolution + ix] = -distance;
      }
    }
  }
  surface.update();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(surface.positionArray.slice(0, surface.count * 3), 3));
  geometry.scale(halfWidth, halfDepth, halfHeight).translate(0, 0, centreZ);
  bodyGeometry = mergeVertices(geometry);
  bodyGeometry.computeVertexNormals();
  bodyGeometry.computeBoundingSphere();
  geometry.dispose();
  surface.geometry.dispose();
  material.dispose();
  return bodyGeometry.clone();
}

const bodyColor = new THREE.Color('#b4bfc3');
const headColor = new THREE.Color('#8ed8c3');
const gaussian = (distance: number, width: number) => Math.exp(-((distance / width) ** 2));

function sculptHead(): THREE.BufferGeometry {
  const rows = 44, sides = 56;
  const positions: number[] = [], indices: number[] = [], colors: number[] = [];
  const tint = new THREE.Color();
  for (let row = 0; row <= rows; row++) {
    const z = THREE.MathUtils.lerp(headProfile[0][0], headProfile.at(-1)![0], row / rows);
    const [, width, depth, , centreY] = profileAt(headProfile, z);
    tint.copy(bodyColor).lerp(headColor, THREE.MathUtils.smoothstep(z, .08, .22));
    for (let side = 0; side < sides; side++) {
      const angle = side * 2 * Math.PI / sides;
      const forward = Math.sin(angle), x = width * Math.cos(angle);
      const face = THREE.MathUtils.smoothstep(forward, .30, .95);
      // Sculpt the nose and closed mouth directly into the surface. Shallow
      // eye contours make the face legible without painted eyes or expression.
      const bridge = .021 * gaussian(x, .019) * gaussian(z - .189, .060);
      const noseTip = .027 * gaussian(x, .024) * gaussian(z - .162, .017);
      const eyes = .010 * gaussian(Math.abs(x) - .042, .026) * gaussian(z - .204, .018);
      const mouth = gaussian(x, .035) * (.005 * gaussian(z - .119, .009) - .004 * gaussian(z - .115, .0035));
      const ear = .010 * gaussian(forward + .04, .27) * gaussian(z - .179, .034);
      positions.push(x + Math.cos(angle) * ear, centreY + depth * forward + face * (bridge + noseTip - eyes + mouth), z);
      colors.push(tint.r, tint.g, tint.b);
      if (row < rows) {
        const a = row * sides + side, b = row * sides + (side + 1) % sides;
        indices.push(a, b, a + sides, b, b + sides, a + sides);
      }
    }
  }
  for (const row of [0, rows]) {
    const centre = positions.length / 3;
    const ring = row === 0 ? headProfile[0] : headProfile.at(-1)!;
    positions.push(0, ring[4], ring[0]);
    const cap = row === 0 ? bodyColor : headColor;
    colors.push(cap.r, cap.g, cap.b);
    for (let side = 0; side < sides; side++) {
      const a = row * sides + side, b = row * sides + (side + 1) % sides;
      if (row === 0) indices.push(centre, b, a);
      else indices.push(centre, a, b);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export function orientationMannequin() {
  const body = new THREE.Mesh(sculptBody(), new THREE.MeshStandardMaterial({ color: bodyColor, roughness: .52 }));
  const head = new THREE.Mesh(sculptHead(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .52 }));
  body.name = 'orientation:body';
  head.name = 'orientation:head';
  const model = new THREE.Group();
  model.add(body, head);
  return { model, body, head };
}
