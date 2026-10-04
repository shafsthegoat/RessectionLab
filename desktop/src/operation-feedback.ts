/** Electron may wrap a sidecar rejection in an IPC prefix. Preserve neutral cancellation. */
export function operationMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
export function isOperationCancelled(error: unknown): boolean {
  return /\boperation cancel(?:led|ed)\b/i.test(operationMessage(error));
}
