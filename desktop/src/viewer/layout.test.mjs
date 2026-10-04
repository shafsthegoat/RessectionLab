import test from "node:test";
import assert from "node:assert/strict";
import { paneViewport } from "./layout.ts";
import {
  PLANE_AXES,
  inverseAffine,
  rasAffine,
  slicePoint,
  sliceRect,
  transformPoint,
  volumeBounds,
} from "./coordinates.ts";

test("Scissor rectangles follow actual image panes and exclude layout controls and headers", () => {
  const root = { left: 40, top: 20, width: 800, height: 640 };
  const axial = { left: 440, top: 20 + 48 + 29, width: 400, height: 251 };
  assert.deepEqual(paneViewport(root, axial), {
    left: 400,
    bottom: 312,
    width: 400,
    height: 251,
  });
  const lower = { left: 40, top: 20 + 48 + 280 + 29, width: 400, height: 251 };
  assert.deepEqual(paneViewport(root, lower), {
    left: 0,
    bottom: 32,
    width: 400,
    height: 251,
  });
  const expanded = { left: 40, top: 20 + 48 + 29, width: 800, height: 531 };
  assert.deepEqual(paneViewport(root, expanded), {
    left: 0,
    bottom: 32,
    width: 800,
    height: 531,
  });
  assert.equal(
    paneViewport(root, { left: 0, top: 0, width: 0, height: 0 }),
    null,
    "Hidden panes must not draw stray pixels",
  );
  assert.equal(
    paneViewport(root, { left: 0, top: 0, width: 400, height: 0 }),
    null,
  );
});

test("The same physical landmark remains clickable in focus, review and expanded MRI panes", () => {
  const angle = Math.PI / 8,
    c = Math.cos(angle),
    s = Math.sin(angle);
  const native = [
    [2 * c, -3 * s, 0, 10],
    [2 * s, 3 * c, 0, -20],
    [0, 0, 4, 7],
    [0, 0, 0, 1],
  ];
  const affine = rasAffine(native, "LPS+"),
    inverse = inverseAffine(affine);
  const bounds = volumeBounds([30, 40, 25], affine),
    sourceVoxel = [12.25, 16.5, 11.75];
  const landmark = transformPoint(affine, sourceVoxel);
  const sizes = [
    [185, 135],
    [225, 220],
    [400, 251],
    [800, 531],
  ];
  for (const plane of ["axial", "coronal", "sagittal"]) {
    const [a, b] = PLANE_AXES[plane];
    for (const [width, height] of sizes) {
      const rect = sliceRect(bounds, plane, width, height);
      const x =
        rect.left +
        ((landmark[a] - bounds[0][a]) / (bounds[1][a] - bounds[0][a])) *
          rect.width;
      const y =
        rect.top +
        ((bounds[1][b] - landmark[b]) / (bounds[1][b] - bounds[0][b])) *
          rect.height;
      const picked = slicePoint(bounds, plane, landmark, rect, x, y);
      picked.forEach((value, axis) =>
        assert.ok(Math.abs(value - landmark[axis]) < 1e-9),
      );
      transformPoint(inverse, picked).forEach((value, axis) =>
        assert.ok(Math.abs(value - sourceVoxel[axis]) < 1e-9),
      );
      assert.ok(
        Math.abs(
          rect.width / (bounds[1][a] - bounds[0][a]) -
            rect.height / (bounds[1][b] - bounds[0][b]),
        ) < 1e-10,
        "Image pixels retain equal physical scale",
      );
    }
  }
});
