import type {
  ArrayDescriptor,
  CasePayload,
  ResectionApi,
  Vec3,
  ViewerCase,
} from "./types";

export function anatomyColor(name: string): string {
  const key = name.toLowerCase();
  if (key.includes("flair")) return "#55bace";
  if (
    key.includes("nonenhanc") ||
    key.includes("necrot") ||
    key.includes("core")
  )
    return "#b894d9";
  if (key.includes("enhanc")) return "#e5a269";
  return ["#91bdad", "#87aee0", "#c2af80"][
    [...key].reduce((sum, c) => sum + c.charCodeAt(0), 0) % 3
  ];
}

export function readableName(name: string): string {
  return name.replace(/_/g, " ").replace(/\bFLAIR\b/i, "FLAIR");
}

export function rasPoint(point: Vec3, frame: string): Vec3 {
  return frame.startsWith("LPS")
    ? [-point[0], -point[1], point[2]]
    : [...point];
}

/** Validate the physical/data contract before fetching or allocating any renderer arrays. */
export function validateCaseDescriptor(payload: CasePayload): {
  voxelCount: number;
  voxelVolumeMm3: number;
} {
  if (!payload || typeof payload.caseId !== "string" || !payload.caseId.trim())
    throw new Error("Missing case identity.");
  if (!/^sha256:[a-f0-9]{64}$/.test(payload.caseHash))
    throw new Error("Missing or invalid case fingerprint.");
  if (payload.frame !== "RAS+" && payload.frame !== "LPS+")
    throw new Error("Unsupported physical coordinate frame.");
  if (
    !Array.isArray(payload.shape) ||
    payload.shape.length !== 3 ||
    payload.shape.some(
      (size) => !Number.isSafeInteger(size) || size < 1 || size > 4096,
    )
  )
    throw new Error("Invalid MRI grid shape.");
  const count = payload.shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(count) || count * 4 > 1024 ** 3)
    throw new Error("MRI exceeds the local viewer size limit.");
  const matrix = payload.affine;
  if (
    !Array.isArray(matrix) ||
    matrix.length !== 4 ||
    matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== 4 ||
        row.some((value) => !Number.isFinite(value)),
    )
  )
    throw new Error("Invalid physical affine.");
  if (
    matrix[3].some(
      (value, index) => Math.abs(value - (index === 3 ? 1 : 0)) > 1e-8,
    )
  )
    throw new Error("The affine is not a homogeneous physical transform.");
  const [a, b, c] = matrix;
  const determinant =
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  const spacing = [0, 1, 2].map((axis) =>
    Math.hypot(a[axis], b[axis], c[axis]),
  );
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= 1e-10 * spacing.reduce((x, y) => x * y, 1)
  )
    throw new Error("The physical affine is singular.");
  if (
    !Array.isArray(payload.spacingMm) ||
    payload.spacingMm.length !== 3 ||
    payload.spacingMm.some(
      (value, index) =>
        !Number.isFinite(value) ||
        value <= 0 ||
        Math.abs(value - spacing[index]) > 1e-5 * Math.max(1, value),
    )
  )
    throw new Error("Voxel spacing disagrees with the physical affine.");
  const validateArray = (
    array: ArrayDescriptor,
    dtype: string,
    name: string,
  ) => {
    if (
      !array ||
      array.dtype !== dtype ||
      array.byteOrder !== "little" ||
      array.order !== "C"
    )
      throw new Error(`${name} has an unsupported array encoding.`);
    if (
      !Array.isArray(array.shape) ||
      array.shape.length !== 3 ||
      array.shape.some((value, index) => value !== payload.shape[index])
    )
      throw new Error(`${name} does not match the MRI grid.`);
    if (array.byteLength !== count * (dtype === "float32" ? 4 : 1))
      throw new Error(`${name} declares an invalid byte length.`);
    if (
      typeof array.assetId !== "string" ||
      !array.assetId ||
      !/^[a-f0-9]{64}$/.test(array.sha256)
    )
      throw new Error(`${name} has an invalid transfer identity.`);
  };
  validateArray(payload.mri, "float32", "MRI");
  if (!Array.isArray(payload.compartments) || payload.compartments.length > 32)
    throw new Error("Invalid target layer inventory.");
  const names = new Set<string>();
  payload.compartments.forEach((layer) => {
    if (
      !layer ||
      typeof layer.name !== "string" ||
      !layer.name ||
      names.has(layer.name) ||
      !Number.isFinite(layer.volumeMm3) ||
      layer.volumeMm3 < 0
    )
      throw new Error("Invalid target layer metadata.");
    names.add(layer.name);
    validateArray(layer.array, "uint8", layer.name);
    if (layer.sourceArray)
      validateArray(layer.sourceArray, "uint8", `${layer.name} source`);
  });
  if (payload.brainMask)
    validateArray(payload.brainMask, "uint8", "Brain mask");
  if (
    !Array.isArray(payload.unknowns) ||
    payload.unknowns.some((item) => typeof item !== "string") ||
    !payload.metadata ||
    typeof payload.metadata !== "object"
  )
    throw new Error("Incomplete case evidence record.");
  if (
    payload.intensityRange &&
    (payload.intensityRange.length !== 2 ||
      payload.intensityRange.some((v) => !Number.isFinite(v)) ||
      payload.intensityRange[0] >= payload.intensityRange[1])
  )
    throw new Error("Invalid MRI intensity range.");
  if (
    payload.targetCentroidMm &&
    (payload.targetCentroidMm.length !== 3 ||
      payload.targetCentroidMm.some((v) => !Number.isFinite(v)))
  )
    throw new Error("Invalid physical target cursor.");
  return { voxelCount: count, voxelVolumeMm3: Math.abs(determinant) };
}

export async function hydrateCase(
  payload: CasePayload,
  api: ResectionApi,
): Promise<ViewerCase> {
  const { voxelVolumeMm3 } = validateCaseDescriptor(payload);
  const [mriBytes, ...maskBytes] = await Promise.all([
    api.readAsset(payload.mri.assetId),
    ...payload.compartments.map((layer) => api.readAsset(layer.array.assetId)),
  ]);
  if (mriBytes.byteLength !== payload.mri.byteLength)
    throw new Error("MRI transfer is incomplete.");
  const mri = new Float32Array(mriBytes.slice().buffer);
  if (mri.some((value) => !Number.isFinite(value)))
    throw new Error("MRI contains non-finite voxel values.");
  const affine = payload.affine.map((row) => [...row]);
  if (payload.frame === "LPS+") {
    affine[0] = affine[0].map((value) => -value);
    affine[1] = affine[1].map((value) => -value);
  }
  return {
    caseId: payload.caseId,
    caseHash: payload.caseHash,
    frame: "RAS+",
    affine,
    shape: [...payload.shape],
    spacingMm: [...payload.spacingMm],
    intensityRange: payload.intensityRange,
    mri,
    compartments: payload.compartments.map((layer, index) => {
      const mask = maskBytes[index];
      if (mask.byteLength !== layer.array.byteLength)
        throw new Error(`Incomplete ${layer.name} transfer.`);
      let selected = 0;
      for (const value of mask) {
        if (value !== 0 && value !== 1)
          throw new Error(`${layer.name} is not a binary mask.`);
        selected += value;
      }
      if (
        Math.abs(selected * voxelVolumeMm3 - layer.volumeMm3) >
        1e-5 * Math.max(1, layer.volumeMm3)
      )
        throw new Error(`${layer.name} volume disagrees with its source grid.`);
      return {
        name: layer.name,
        volumeMm3: layer.volumeMm3,
        mask,
        color: anatomyColor(layer.name),
      };
    }),
  };
}

export function initialCursor(caseData: ViewerCase): Vec3 {
  const [sx, sy, sz] = caseData.shape;
  let count = 0,
    x = 0,
    y = 0,
    z = 0;
  for (const layer of caseData.compartments) {
    for (let index = 0; index < layer.mask.length; index++) {
      if (!layer.mask[index]) continue;
      count++;
      x += Math.floor(index / (sy * sz));
      y += Math.floor(index / sz) % sy;
      z += index % sz;
    }
  }
  const point = count
    ? [x / count, y / count, z / count]
    : [(sx - 1) / 2, (sy - 1) / 2, (sz - 1) / 2];
  return caseData.affine
    .slice(0, 3)
    .map(
      (row) =>
        row[0] * point[0] + row[1] * point[1] + row[2] * point[2] + row[3],
    ) as Vec3;
}
