import { maskSurface } from "./surface";
import { surfaceNormals } from "./surfaceNormals";
import { smoothSurface } from "./surfaceSmoothing";
import type { Affine, Shape3 } from "./coordinates";

self.onmessage = (
  event: MessageEvent<{ name: string; mask: Uint8Array; shape: Shape3; displaySmoothing?: boolean; affine?: Affine }>,
) => {
  const { name, mask, shape, displaySmoothing, affine } = event.data;
  try {
    const positions = maskSurface(mask, shape);
    const normals = surfaceNormals(positions);
    if (displaySmoothing) {
      if (!affine) throw new Error("Display smoothing requires the source affine.");
      const smoothPositions = smoothSurface(positions, affine);
      const smoothNormals = surfaceNormals(smoothPositions);
      self.postMessage(
        { name, positions, normals, smoothPositions, smoothNormals },
        { transfer: [positions.buffer, normals.buffer, smoothPositions.buffer, smoothNormals.buffer] },
      );
      return;
    }
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
