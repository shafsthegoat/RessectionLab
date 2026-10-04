import { Brain, CircleDot } from "lucide-react";
import type { StructuralEvidence } from "./types";

export function StructuralEvidenceInventory({
  evidence,
  detailed = false,
}: {
  evidence: StructuralEvidence[];
  detailed?: boolean;
}) {
  if (!evidence.length) return null;
  return (
    <section
      className={
        detailed
          ? "record-section structural-section"
          : "case-section structural-section"
      }
    >
      <h2>
        Structural proposals{" "}
        <span>{evidence.length.toString().padStart(2, "0")}</span>
      </h2>
      {evidence.map((item) => {
        const variant = item.metadata?.variant;
        const label =
          item.kind === "whole_brain_envelope"
            ? variant === "nocsf"
              ? "Brain envelope · no CSF"
              : "Brain envelope"
            : item.kind.replace(/_/g, " ");
        const state =
          item.reviewStatus === "rejected"
            ? "Rejected"
            : item.reviewRequired
              ? "Review required"
              : item.reviewStatus === "accepted"
                ? "Reviewed envelope"
                : "Review unassessed";
        const outside = item.metadata?.current_target_annotation_outside_voxels;
        return (
          <div className="structural-proposal" key={item.evidenceId}>
            <div className="structural-proposal-heading">
              <Brain size={15} />
              <strong>{label}</strong>
            </div>
            <div className="structural-proposal-state">
              <span>
                {item.provenance === "estimated"
                  ? "Model estimate"
                  : item.provenance}
              </span>
              <span className="unassessed-pill">{state}</span>
            </div>
            {typeof outside === "number" && outside > 0 && (
              <p className="structural-qc">
                <CircleDot size={10} />
                {outside.toLocaleString()} supplied annotation voxels lie
                outside this estimate.
              </p>
            )}
            {detailed && (
              <>
                <p className="structural-method">{item.method}</p>
                <dl>
                  <dt>Model identity</dt>
                  <dd>
                    <code title={item.modelHash}>
                      {item.modelHash.slice(0, 22)}…
                    </code>
                  </dd>
                  <dt>Source identity</dt>
                  <dd>
                    <code title={item.sourceHash}>
                      {item.sourceHash.slice(0, 22)}…
                    </code>
                  </dd>
                  <dt>Cortical access</dt>
                  <dd>Not certified</dd>
                </dl>
              </>
            )}
          </div>
        );
      })}
      <p className="muted-note">
        Stored separately from working anatomy. A whole-brain envelope does not
        certify cortex or permit cortical access.
      </p>
    </section>
  );
}
