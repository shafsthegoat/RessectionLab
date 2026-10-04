import { useMemo } from "react";
import { ChevronDown, Globe2, LoaderCircle, X } from "lucide-react";
import { priorLabel, samplePriorAtCursor } from "./prior-data";
import { formatPriorValue } from "./viewer/priorLayer";
import type { PriorLayerView, PriorProposal, Vec3 } from "./types";

export function PriorCursorReadout({
  layer,
  cursor,
}: {
  layer: PriorLayerView;
  cursor: Vec3 | null;
}) {
  const sample = useMemo(
    () => samplePriorAtCursor(layer, cursor),
    [layer, cursor],
  );
  const functional = layer.mapKind === "functional_concordance";
  return (
    <div className="prior-cursor-readout">
      <span className="eyebrow">
        AT LINKED CURSOR ·{" "}
        {functional
          ? "INTERPOLATED ATLAS SAMPLE"
          : "NEAREST RELEASED-MASK CELL"}
      </span>
      <strong>
        {sample.covered
          ? "Within atlas field of view"
          : sample.reason === "numerical-boundary-uncertainty"
            ? "Atlas field boundary — display precision unknown"
            : sample.reason === "numerical-precision-unavailable"
              ? "Atlas coordinates exceed display precision — unknown"
              : sample.reason === "incomplete-interpolation-support"
                ? "Incomplete atlas sampling support — unknown"
                : "Outside atlas field of view — unknown"}
      </strong>
      {sample.covered && (
        <>
          <div className="prior-cursor-value">
            <span>
              {functional ? "Atlas concordance" : "Released mask membership"}
            </span>
            <b>
              {functional
                ? formatPriorValue(sample.value!)
                : sample.value === 1
                  ? "Included (1)"
                  : "Not included (0)"}
            </b>
          </div>
          <p>
            {sample.value === 0
              ? "No signal in this released map; patient function unknown."
              : "Population map value; patient function remains unknown."}
          </p>
        </>
      )}
      {!sample.covered && (
        <p>
          No complete atlas sample is available here. Patient function is
          unknown.
        </p>
      )}
    </div>
  );
}
export function PriorInventory({
  items,
  selected,
  loadingId,
  disabled,
  onSelect,
  onClear,
  cursor,
}: {
  items: PriorProposal[];
  selected: PriorLayerView | null;
  loadingId: string | null;
  disabled: boolean;
  onSelect: (id: string) => void;
  onClear: () => void;
  cursor: Vec3 | null;
}) {
  if (!items.length) return null;
  return (
    <section className="case-section prior-inventory">
      <h2>
        Population priors <span>{String(items.length).padStart(2, "0")}</span>
      </h2>
      <div className="prior-population-badge">
        <Globe2 size={12} />
        Population prior
      </div>
      <p className="prior-review-note">
        Alignment review required · inspection only
      </p>
      <label className="field-label" htmlFor="prior-layer">
        Inspect one released map
      </label>
      <div className="select-wrap">
        <select
          id="prior-layer"
          value={loadingId ?? selected?.proposalId ?? ""}
          disabled={disabled}
          onChange={(event) =>
            event.target.value ? onSelect(event.target.value) : onClear()
          }
        >
          <option value="">Source only · no prior overlay</option>
          {items.map((item) => (
            <option key={item.proposalId} value={item.proposalId}>
              {priorLabel(item)}
            </option>
          ))}
        </select>
        <ChevronDown size={13} />
      </div>
      {loadingId && (
        <button className="text-button" onClick={onClear}>
          <LoaderCircle size={12} className="spin" />
          Cancel loading prior
        </button>
      )}
      {selected && (
        <>
          <div className="prior-value-legend">
            {selected.mapKind === "functional_concordance" ? (
              <>
                <strong>Atlas concordance (unitless, 0–1)</strong>
                <div className="prior-concordance-scale" />
                <div className="prior-scale-ticks">
                  <span>0</span>
                  <span>1</span>
                </div>
                <p>
                  Normative connectivity-map concordance. This is not an injury
                  probability or patient connectivity.
                </p>
              </>
            ) : (
              <>
                <strong>Released mask membership (0 or 1)</strong>
                <p>
                  Included in the released normative mask. This is not a patient
                  tract or tract probability.
                </p>
              </>
            )}
          </div>
          <PriorCursorReadout layer={selected} cursor={cursor} />
          <button className="text-button" onClick={onClear}>
            <X size={12} />
            Return to source view
          </button>
        </>
      )}
      <div className="prior-use-note">
        <strong>Not used in route scoring</strong>
        <span>Patient language dominance: unknown</span>
        <span>Patient-specific function: unknown</span>
      </div>
    </section>
  );
}
export function PriorProvenance({ proposal }: { proposal: PriorProposal }) {
  const record = proposal.provenanceRecord;
  return (
    <section className="record-section prior-provenance">
      <h3>{priorLabel(proposal)}</h3>
      <div className="prior-population-badge">
        <Globe2 size={12} />
        Population prior · inspection only
      </div>
      <dl>
        <dt>Review</dt>
        <dd>Alignment review required</dd>
        <dt>Source category</dt>
        <dd>
          {String(proposal.metadata.source_category ?? proposal.component)}
        </dd>
        <dt>Map values</dt>
        <dd>
          {proposal.mapKind === "functional_concordance"
            ? "Atlas concordance (unitless, 0–1)"
            : "Released mask membership (0 or 1)"}
        </dd>
        <dt>Spatial units</dt>
        <dd>mm</dd>
        <dt>Atlas field-of-view coverage</dt>
        <dd>
          {(proposal.samplingCoverageFraction * 100).toFixed(2)}% of the full
          image box, including background. This is not patient functional
          coverage or confidence.
        </dd>
        <dt>Interpolation</dt>
        <dd>{proposal.interpolation.replace(/_/g, " ")}</dd>
        <dt>Transform direction</dt>
        <dd>Template RAS mm → patient RAS mm</dd>
        <dt>Method / version</dt>
        <dd>
          {String(record.registration_method ?? "Unrecorded")} /{" "}
          {String(record.registration_version ?? "Unrecorded")}
        </dd>
        <dt>Source member hash</dt>
        <dd>
          <code>{proposal.source.sha256}</code>
        </dd>
        <dt>Registration hash</dt>
        <dd>
          <code>{proposal.registrationHash}</code>
        </dd>
        <dt>Review scope</dt>
        <dd>Inspection only · not used in route scoring</dd>
      </dl>
      <p className="muted-note">
        {String(
          proposal.metadata.template_redistribution ??
            "Source template is not embedded in this view.",
        )}
      </p>
    </section>
  );
}
