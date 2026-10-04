import { maskSurface } from "./surface";
import type { Shape3 } from "./coordinates";

self.onmessage = (
  event: MessageEvent<{ name: string; mask: Uint8Array; shape: Shape3 }>,
) => {
  const { name, mask, shape } = event.data;
  try {
    const positions = maskSurface(mask, shape);
    self.postMessage({ name, positions }, { transfer: [positions.buffer] });
  } catch (error) {
    self.postMessage({
      name,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
