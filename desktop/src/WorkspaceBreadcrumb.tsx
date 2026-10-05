import { ChevronRight } from "lucide-react";

export function WorkspaceBreadcrumb({ caseId }: { caseId: string | null }) {
  return (
    <div className="workspace-breadcrumb">
      <span className="workspace-context">Workspace</span>
      <ChevronRight size={12} aria-hidden="true" />
      {caseId ? (
        <span
          className="workspace-case-id"
          role="status"
          aria-label={`Current case: ${caseId}`}
          title={`Current case: ${caseId}`}
        >
          {caseId}
        </span>
      ) : (
        <span>Route comparison</span>
      )}
    </div>
  );
}
