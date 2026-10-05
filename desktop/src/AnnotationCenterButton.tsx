import { Focus } from "lucide-react";
import type { Vec3 } from "./types";

export function AnnotationCenterButton({
  center,
  onCenter,
}: {
  center: Vec3 | null;
  onCenter: (point: Vec3) => void;
}) {
  return (
    <button
      type="button"
      className="text-button annotation-center-button"
      disabled={!center}
      title={
        center
          ? "Move the linked MRI slices to the source annotation center."
          : "Open a case with target annotations to center its MRI slices."
      }
      onClick={() => {
        if (center) onCenter([...center]);
      }}
    >
      <Focus size={13} aria-hidden="true" />
      Center on annotations
    </button>
  );
}
