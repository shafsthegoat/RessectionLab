import { ChevronDown, FlaskConical, LoaderCircle } from "lucide-react";
import {
  neighboringPathReason,
  neighboringPathsRequest,
  neighboringToolChoices,
  validateNeighboringPaths,
} from "./neighboring-paths";
import type {
  NeighboringPathsChoices,
  NeighboringPathsContext,
  NeighboringPathsRequest,
  NeighboringPathsView,
} from "./neighboring-paths";
import "./NeighboringPathsPanel.css";

export type NeighboringPathsState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "cancelled" }
  | { kind: "error"; message: string }
  | { kind: "complete"; result: unknown };

export interface NeighboringPathsPanelProps {
  context: NeighboringPathsContext | null;
  choices: NeighboringPathsChoices;
  state: NeighboringPathsState;
  unavailableReason?: string;
  onChoices: (choices: NeighboringPathsChoices) => void;
  onInspect: (request: NeighboringPathsRequest) => void;
  onCancel: () => void;
}

const coordinates = (point: number[]) =>
  point.map((n) => n.toFixed(2)).join(" / ");
const toolName = (id: string) =>
  neighboringToolChoices.find((tool) => tool.id === id)?.label ?? id;
const statusLabel = {
  preview_passed: "Preview passed",
  rejected: "Preview rejected",
  omitted: "Omitted",
};

/** Controlled, view-only panel: rendering and acknowledgment never launch work. */
export function NeighboringPathsPanel({
  context,
  choices,
  state,
  unavailableReason,
  onChoices,
  onInspect,
  onCancel,
}: NeighboringPathsPanelProps) {
  const request = neighboringPathsRequest(context, choices);
  const loading = state.kind === "loading";
  const blocked =
    loading || !context || context.support === "blocked" || !!unavailableReason;
  let view: NeighboringPathsView | null = null,
    withheld: string | null = null;
  if (state.kind === "complete") {
    try {
      if (unavailableReason)
        throw new Error(
          "Inspection is currently unavailable. Previous results are withheld.",
        );
      if (!context || !request)
        throw new Error(
          "Inspection assumptions changed. Choose the settings and inspect again.",
        );
      view = validateNeighboringPaths(state.result, request, context);
    } catch (error) {
      withheld =
        error instanceof Error
          ? error.message
          : "Inspection record could not be checked.";
    }
  }
  return (
    <section
      className="neighboring-paths-panel"
      aria-labelledby="neighboring-paths-title"
    >
      <div className="planning-title">
        <span className="eyebrow">
          <FlaskConical size={12} aria-hidden="true" /> Experimental · view only
        </span>
        <h2 id="neighboring-paths-title">Inspect neighboring paths</h2>
        <p>
          The selected route supplies an access window only. These samples can
          change the entry and endpoint; they do not refine the fixed route.
        </p>
      </div>
      <p className="neighboring-anchor">
        Access window from{" "}
        <strong>{context?.routeLabel ?? "Choose a source route"}</strong>
      </p>
      <fieldset className="neighboring-tools" disabled={blocked}>
        <legend>Research instruments</legend>
        <p>
          Choose one or both. Their geometry is separate from the selected
          route’s instrument.
        </p>
        {neighboringToolChoices.map((tool) => (
          <label key={tool.id}>
            <input
              type="checkbox"
              checked={choices.toolIds.includes(tool.id)}
              disabled={
                !context?.toolCatalog.some((item) => item.tool_id === tool.id)
              }
              onChange={(event) =>
                onChoices({
                  ...choices,
                  toolIds: event.target.checked
                    ? [...choices.toolIds, tool.id]
                    : choices.toolIds.filter((id) => id !== tool.id),
                })
              }
            />
            {tool.label}
          </label>
        ))}
      </fieldset>
      <label className="neighboring-ack">
        <input
          type="checkbox"
          checked={choices.neighboringColumns}
          disabled={blocked}
          onChange={(event) =>
            onChoices({ ...choices, neighboringColumns: event.target.checked })
          }
        />
        <span>
          Inspect neighboring entries and endpoints using this access window.
        </span>
      </label>
      {context?.support === "acknowledgment_required" && (
        <label className="neighboring-ack">
          <input
            type="checkbox"
            checked={choices.estimatedSupport}
            disabled={blocked}
            onChange={(event) =>
              onChoices({ ...choices, estimatedSupport: event.target.checked })
            }
          />
          <span>
            Use the source-bound estimated tissue support for this inspection.
            It does not establish cortical access.
          </span>
        </label>
      )}
      {context?.supportReason && (
        <p className="capability-note">{context.supportReason}</p>
      )}
      {context?.support === "blocked" && !context.supportReason && (
        <p className="capability-note">
          Access support needs review before this inspection is available. An
          estimated envelope does not establish cortical access.
        </p>
      )}
      {!context && (
        <p className="capability-note">
          Choose a source route to identify its hypothetical access window.
        </p>
      )}
      {unavailableReason && (
        <p className="capability-note">{unavailableReason}</p>
      )}
      <button
        type="button"
        className="outline-button neighboring-inspect"
        disabled={blocked || !request}
        onClick={() => {
          if (!blocked && request) onInspect(request);
        }}
      >
        {loading ? (
          <LoaderCircle size={14} className="spin" aria-hidden="true" />
        ) : (
          <FlaskConical size={14} aria-hidden="true" />
        )}
        {loading ? "Inspecting initial paths…" : "Inspect neighboring paths"}
      </button>
      {loading && (
        <>
          <p className="neighboring-status" role="status">
            Checking the full declared sample. No partial inventory is shown.
          </p>
          <button type="button" className="text-button" onClick={onCancel}>
            Cancel inspection
          </button>
        </>
      )}
      {state.kind === "cancelled" && (
        <p className="neighboring-status" role="status">
          Inspection cancelled; no result published. A geometry check may still
          be finishing.
        </p>
      )}
      {state.kind === "error" && (
        <div className="capability-note" role="alert">
          <strong>Inspection unavailable</strong>
          <p>{neighboringPathReason(state.message)}</p>
          <p>
            The selected window stays unchanged; no replacement direction is
            chosen.
          </p>
        </div>
      )}
      {withheld && (
        <div className="capability-note" role="alert">
          <strong>Inspection result withheld</strong>
          <p>{withheld}</p>
        </div>
      )}
      {view && (
        <section
          className="neighboring-results"
          aria-label="Initial neighboring-path inventory"
        >
          <h3>Initial sampled paths</h3>
          <p className="neighboring-status" role="status">
            {view.total} of {view.total} declared path/tool pairs accounted for
            · 13 neighboring columns × {view.total / 13} instrument
            {view.total / 13 === 1 ? "" : "s"}.
          </p>
          <p>
            This is a bounded initial sample, not an exhaustive route search.
          </p>
          <dl className="neighboring-counts">
            <div>
              <dt>Preview passed</dt>
              <dd>{view.passed}</dd>
            </div>
            <div>
              <dt>Preview rejected</dt>
              <dd>{view.rejected}</dd>
            </div>
            <div>
              <dt>Omitted before preview</dt>
              <dd>{view.omitted}</dd>
            </div>
            <div>
              <dt>Geometry checks, including fallback</dt>
              <dd>{view.previewAttempts}</dd>
            </div>
          </dl>
          {!view.passed && (
            <p className="capability-note">
              No sampled path passed the initial preview. Omitted and rejected
              paths remain listed below.
            </p>
          )}
          <p>No actions executed · no tissue changed.</p>
          <div className="neighboring-ledger">
            {view.rows.map((row) => (
              <details key={row.ordinal} className="neighboring-path-row">
                <summary>
                  <span>
                    Path {String(row.pathNumber).padStart(2, "0")} ·{" "}
                    {toolName(row.toolId)}
                  </span>
                  <span>{statusLabel[row.status]}</span>
                  <ChevronDown size={12} aria-hidden="true" />
                </summary>
                <p>{neighboringPathReason(row.reason)}</p>
                {row.entryMm ? (
                  <>
                    <p
                      className="neighboring-coordinates"
                      title={`Entry RAS mm: ${row.entryMm.join(" / ")}`}
                    >
                      Entry · RAS mm
                      <br />
                      {coordinates(row.entryMm)}
                    </p>
                    {row.attempts.map((attempt) => (
                      <div className="neighboring-attempt" key={attempt.phase}>
                        <strong>
                          {attempt.phase === "primary"
                            ? "Primary endpoint"
                            : "Fallback after primary rejection"}{" "}
                          ·{" "}
                          {attempt.passed
                            ? "preview passed"
                            : "preview rejected"}
                        </strong>
                        <p
                          className="neighboring-coordinates"
                          title={`Requested endpoint RAS mm: ${attempt.endpointMm.join(" / ")}`}
                        >
                          {coordinates(attempt.endpointMm)} · RAS mm
                        </p>
                        <p>{neighboringPathReason(attempt.reason)}</p>
                        {!attempt.passed && (
                          <p>
                            Requested endpoint shown; the first failure location
                            is unavailable.
                          </p>
                        )}
                      </div>
                    ))}
                  </>
                ) : (
                  <p>No endpoint generated for this omitted path.</p>
                )}
              </details>
            ))}
          </div>
          <details className="neighboring-binding">
            <summary>
              Unassessed evidence ({view.unknowns.length})
              <ChevronDown size={12} aria-hidden="true" />
            </summary>
            <ul>
              {view.unknowns.map((item, index) => (
                <li key={`${index}:${item}`} title={item}>
                  {neighboringPathReason(item)}
                </li>
              ))}
            </ul>
          </details>
          <details className="neighboring-binding">
            <summary>
              Instrument geometry and inspection assumptions{" "}
              <ChevronDown size={12} aria-hidden="true" />
            </summary>
            <p>
              Coordinates use RAS millimeters. Neighbor offsets in this record
              are source voxels. The three-step configured horizon was not
              executed; only the initial state was inspected.
            </p>
            <pre>{JSON.stringify(view.binding, null, 2)}</pre>
          </details>
        </section>
      )}
      <div className="neighboring-limits">
        <strong>Inspection only</strong>
        <p>
          Motor, language and vascular anatomy remain unassessed. Displayed
          population priors are not used. Clinical outcomes are not estimated.
        </p>
        <p>
          Instrument portions outside the source image remain unassessed. These
          previews do not authorize a candidate route or tissue removal.
        </p>
      </div>
    </section>
  );
}
