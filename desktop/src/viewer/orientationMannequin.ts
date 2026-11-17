import * as THREE from 'three';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

// Unitless sculpt in patient coordinates, with the neck pivot at the origin.
// +X is right, +Y is anterior, and +Z is superior.
type Ring = [z: number, halfWidth: number, halfDepth: number, x: number, y: number];

// Cross-sections ascend from the pelvis through the waist, ribcage and collar.
const torso: Ring[] = [
  [-.795, .041, .049, 0, -.004], [-.747, .119, .088, 0, -.014],
  [-.670, .167, .106, 0, -.016], [-.608, .161, .097, 0, -.012],
  [-.540, .123, .078, 0, -.009], [-.474, .120, .080, 0, -.007],
  [-.390, .145, .097, 0, -.004], [-.308, .176, .113, 0, .005],
  [-.224, .190, .110, 0, .001], [-.162, .186, .093, 0, -.009],
  [-.106, .157, .071, 0, -.013], [-.054, .098, .065, 0, -.013],
  [-.020, .073, .051, 0, -.011], [.021, .060, .048, 0, -.011],
];
// The hand flares below a thin wrist. The forearm changes direction at the
// elbow, and the deltoid has its own rounded peak above the upper arm.
const arm: Ring[] = [
  [-.848, .012, .010, .339, .036], [-.821, .028, .018, .340, .030],
  [-.773, .035, .025, .336, .023], [-.721, .022, .022, .323, .012],
  [-.662, .027, .030, .313, .005], [-.585, .041, .040, .303, -.004],
  [-.535, .040, .040, .295, -.020], [-.490, .031, .035, .287, -.031],
  [-.442, .040, .044, .273, -.018], [-.369, .048, .051, .256, -.004],
  [-.304, .045, .050, .241, -.003], [-.246, .058, .057, .225, -.006],
  [-.194, .063, .059, .211, -.008], [-.144, .046, .043, .196, -.010],
  [-.115, .018, .020, .176, -.012],
];
// The knee leads the calf in profile. The ankle sits above the rear of the
// foot, whose flatter sole and dropping forefoot create the wedge.
const leg: Ring[] = [
  [-1.535, .033, .086, .119, .049], [-1.518, .046, .109, .119, .058],
  [-1.477, .043, .091, .116, .044], [-1.440, .035, .060, .113, .018],
  [-1.391, .027, .032, .111, -.008], [-1.337, .030, .039, .108, -.015],
  [-1.245, .051, .061, .106, -.030], [-1.184, .050, .055, .103, -.023],
  [-1.131, .038, .042, .097, -.007], [-1.081, .044, .050, .099, .011],
  [-1.025, .050, .055, .097, .002], [-.959, .062, .067, .092, -.003],
  [-.856, .079, .085, .090, -.003], [-.770, .084, .098, .087, -.010],
  [-.690, .080, .087, .081, -.012], [-.619, .034, .042, .076, -.014],
];
const headProfile: Ring[] = [
  [-.044, .059, .049, 0, -.010], [-.013, .060, .051, 0, -.010],
  [.007, .056, .054, 0, -.006], [.025, .044, .067, 0, .011],
  [.049, .059, .079, 0, .010], [.077, .074, .082, 0, .004],
  [.115, .084, .090, 0, -.002], [.154, .089, .092, 0, -.011],
  [.196, .092, .099, 0, -.015], [.244, .085, .094, 0, -.016],
  [.278, .067, .076, 0, -.018], [.300, .038, .045, 0, -.018],
  [.310, .001, .001, 0, -.017],
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

function profileDistance(profile: Ring[], ring: Ring, x: number, y: number, squareness = 0): number {
  const [z, width, depth, centreX, centreY] = ring;
  const u = (x - centreX) / width, v = (y - centreY) / depth;
  // A rounded rectangular cross-section gives the chest and pelvis broad
  // planes, and gives the foot a sole that reads from above as well as the side.
  const ellipse = Math.hypot(u, v);
  const radius = squareness > 0 ? THREE.MathUtils.lerp(ellipse, Math.sqrt(Math.hypot(u * u, v * v)), squareness) : ellipse;
  const side = (radius - 1) * Math.min(width, depth);
  return Math.max(side, profile[0][0] - z, z - profile[profile.length - 1][0]);
}

function blend(a: number, b: number, radius: number): number {
  const overlap = Math.max(radius - Math.abs(a - b), 0) / radius;
  return Math.min(a, b) - overlap * overlap * radius * .25;
}

function thumbDistance(x: number, y: number, z: number): number {
  const offset = z + .779;
  const centreX = .300 - .24 * offset;
  return (Math.hypot((x - centreX) / .020, (y - .036) / .018, offset / .041) - 1) * .018;
}

let bodyGeometry: THREE.BufferGeometry | undefined;
function sculptBody(): THREE.BufferGeometry {
  if (bodyGeometry) return bodyGeometry.clone();

  // Keep the narrow seams at the armpits and groin while blending the joints
  // themselves into one molded surface.
  const resolution = 50, maxTriangles = 18000;
  const material = new THREE.MeshBasicMaterial();
  const surface = new MarchingCubes(resolution, material, false, false, maxTriangles);
  surface.isolation = 0;
  const halfWidth = .43, halfDepth = .22, halfHeight = .88, centreZ = -.75;
  const axis = (index: number) => index * 2 / resolution - 1;
  for (let iz = 0; iz < resolution; iz++) {
    const z = axis(iz) * halfHeight + centreZ;
    const torsoRing = profileAt(torso, z), armRing = profileAt(arm, z), legRing = profileAt(leg, z);
    const footShape = 1 - THREE.MathUtils.smoothstep(z, -1.48, -1.40);
    for (let iy = 0; iy < resolution; iy++) {
      const y = axis(iy) * halfDepth;
      for (let ix = 0; ix < resolution; ix++) {
        const x = Math.abs(axis(ix) * halfWidth);
        const core = profileDistance(torso, torsoRing, x, y, .70);
        const armDistance = blend(profileDistance(arm, armRing, x, y), thumbDistance(x, y, z), .014);
        const legDistance = profileDistance(leg, legRing, x, y, footShape);
        const distance = blend(blend(core, armDistance, .026), legDistance, .028);
        surface.field[(iz * resolution + iy) * resolution + ix] = -distance;
      }
    }
  }
  surface.update();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(surface.positionArray.slice(0, surface.count * 3), 3));
  geometry.scale(halfWidth, halfDepth, halfHeight).translate(0, 0, centreZ);
  bodyGeometry = mergeVertices(geometry);
  // An isovalue that meets a grid corner can emit a face that welds to a line.
  const welded = bodyGeometry.getIndex()!;
  const triangles: number[] = [];
  for (let i = 0; i < welded.count; i += 3) {
    const a = welded.getX(i), b = welded.getX(i + 1), c = welded.getX(i + 2);
    if (a !== b && b !== c && a !== c) triangles.push(a, b, c);
  }
  bodyGeometry.setIndex(triangles);
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
    tint.copy(bodyColor).lerp(headColor, THREE.MathUtils.smoothstep(z, .012, .152));
    for (let side = 0; side < sides; side++) {
      const angle = side * 2 * Math.PI / sides;
      const forward = Math.sin(angle), x = width * Math.cos(angle);
      const face = THREE.MathUtils.smoothstep(forward, .30, .95);
      // Chin and brow establish a profile. A shallow ridge keeps the front
      // recognizable while the face stays smooth and expressionless.
      const chin = .012 * gaussian(x, .051) * gaussian(z - .043, .019);
      const nose = .026 * gaussian(x, .023) * gaussian(z - .109, .031);
      const forehead = .008 * gaussian(x, .070) * gaussian(z - .184, .047);
      const ear = .008 * gaussian(forward + .12, .24) * gaussian(z - .125, .031);
      positions.push(x + Math.cos(angle) * ear, centreY + depth * forward + face * (chin + nose + forehead), z);
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
