import { maskSurface } from "./surface";
import { surfaceNormals } from "./surfaceNormals";
import type { Shape3 } from "./coordinates";

self.onmessage = (
  event: MessageEvent<{ name: string; mask: Uint8Array; shape: Shape3 }>,
) => {
  const { name, mask, shape } = event.data;
  try {
    const positions = maskSurface(mask, shape);
    const normals = surfaceNormals(positions);
    self.postMessage(
      { name, positions, normals },
      { transfer: [positions.buffer, normals.buffer] },
    );
  } catch (error) {
    self.postMessage({
      name,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
