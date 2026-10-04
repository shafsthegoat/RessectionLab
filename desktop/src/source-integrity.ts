import type { CasePayload, ViewerCase, Vec3 } from "./types";
const sourceDigests = new WeakMap<ViewerCase, Promise<string>>();
const hex = (bytes: ArrayBuffer) =>
  Array.from(new Uint8Array(bytes), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
export async function sha256Bytes(bytes: Uint8Array): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", bytes.slice().buffer));
}
export function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted)
    throw new DOMException("Proposal viewing cancelled.", "AbortError");
}
/** Match the existing Python source-array digest header; array values are never resampled. */
export async function arrayDigest(
  bytes: Uint8Array,
  shape: Vec3,
  dtype: "<f4" | "|b1",
): Promise<string> {
  const header = new TextEncoder().encode(
    `{"dtype": "${dtype}", "shape": [${shape.join(", ")}]}`,
  );
  const framed = new Uint8Array(header.length + bytes.byteLength);
  framed.set(header);
  framed.set(bytes, header.length);
  return `sha256:${await sha256Bytes(framed)}`;
}
/** Python JSON retains .0 on floats and uses scientific notation outside this interval. */
export function pythonAffineFloat(value: number): string {
  if (!Number.isFinite(value))
    throw new Error("Non-finite proposal source frame.");
  if (Object.is(value, -0)) return "-0.0";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    const [fraction, exponent] = value.toExponential().split("e");
    const power = Number(exponent);
    return `${fraction}e${power < 0 ? "-" : "+"}${Math.abs(power).toString().padStart(2, "0")}`;
  }
  return Number.isInteger(value) ? `${value}.0` : value.toString();
}
export async function sourceFrameDigest(source: CasePayload): Promise<string> {
  const affine = `[${source.affine.map((row) => `[${row.map(pythonAffineFloat).join(",")}]`).join(",")}]`;
  const text = `{"affine":${affine},"frame":"${source.frame}","physical_units":"mm","shape":[${source.shape.join(",")}]}`;
  return `sha256:${await sha256Bytes(new TextEncoder().encode(text))}`;
}

export function sourceImageDigest(
  viewer: ViewerCase,
  shape: Vec3,
): Promise<string> {
  let pending = sourceDigests.get(viewer);
  if (!pending) {
    pending = arrayDigest(
      new Uint8Array(
        viewer.mri.buffer,
        viewer.mri.byteOffset,
        viewer.mri.byteLength,
      ),
      shape,
      "<f4",
    );
    sourceDigests.set(viewer, pending);
  }
  return pending;
}
