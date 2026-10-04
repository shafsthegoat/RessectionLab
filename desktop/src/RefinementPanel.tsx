import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Check,
  ChevronDown,
  FlaskConical,
  LoaderCircle,
  Play,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import type {
  BridgeEvent,
  CasePayload,
  ResectionApi,
  RouteCandidate,
} from "./types";

import { hydrateCertifiedReplay } from "./training-data";
import type { CertifiedReplay, TrainingRun, TrainingStats } from "./types";

function SelectionCurve({
  points,
}: {
  points: { gradient_steps: number; mean_return: number }[];
}) {
  const data = points.filter(
    (point) =>
      Number.isFinite(point.mean_return) &&
      Number.isFinite(point.gradient_steps),
  );
  if (!data.length) return null;
  const low = Math.min(...data.map((point) => point.mean_return)),
    high = Math.max(...data.map((point) => point.mean_return));
  const maxStep = Math.max(1, ...data.map((point) => point.gradient_steps));
  const coordinates = data.map((point) => [
    25 + (point.gradient_steps / maxStep) * 235,
    78 - ((point.mean_return - low) / Math.max(1, high - low)) * 52,
  ]);
  return (
    <figure className="selection-curve">
      <figcaption>
        Selection-world return <span>Surrogate objective</span>
      </figcaption>
      <svg
        viewBox="0 0 280 108"
        role="img"
        aria-label={`Selection return at ${data.length} checkpoints, ranging from ${low.toFixed(2)} to ${high.toFixed(2)}`}
      >
        <path d="M25 18V85H267" fill="none" stroke="#334c58" />
        <path
          d={coordinates
            .map(
              (point, index) => `${index ? "L" : "M"}${point[0]},${point[1]}`,
            )
            .join(" ")}
          fill="none"
          stroke="#91cabc"
          strokeWidth="1.6"
        />
        {coordinates.map((point, index) => (
          <circle
            key={index}
            cx={point[0]}
            cy={point[1]}
            r="2.5"
            fill="#b8e8d9"
          />
        ))}
        <text x="25" y="101">
          0 updates
        </text>
        <text x="260" y="101" textAnchor="end">
          {maxStep} updates
        </text>
        <text x="25" y="11">
          {high.toFixed(1)}
        </text>
      </svg>
      <p>
        Selection is used to choose a checkpoint. Final evaluation remains
        untouched.
      </p>
    </figure>
  );
}

export function RefinementPanel({
  api,
  caseData,
  route,
  busy,
  onReplay,
  onError,
}: {
  api: ResectionApi | null;
  caseData: CasePayload | null;
  route: RouteCandidate | undefined;
  busy: boolean;
  onReplay: (replay: CertifiedReplay | null) => void;
  onError: (error: string) => void;
}) {
  const [supported, setSupported] = useState(false),
    [budget, setBudget] = useState(30),
    [seed, setSeed] = useState(0);
  const [runs, setRuns] = useState<TrainingRun[]>([]),
    [run, setRun] = useState<TrainingRun | null>(null),
    [live, setLive] = useState<TrainingStats | null>(null);
  const [working, setWorking] = useState(false),
    [note, setNote] = useState(""),
    [step, setStep] = useState(0),
    [replay, setReplay] = useState<CertifiedReplay | null>(null);
  const currentHash = useRef(caseData?.caseHash);
  currentHash.current = caseData?.caseHash;
  const replayGeneration = useRef(0);
  const blocked = busy || working,
    readonly = !!api?.readOnly;
  const stats = run?.training ?? live;
  const accepted =
    !!stats?.replay && stats.replay_status === "accepted_independent_geometry";
  useEffect(() => {
    let disposed = false;
    setSupported(false);
    setRun(null);
    setRuns([]);
    setLive(null);
    setReplay(null);
    setNote("");
    replayGeneration.current++;
    onReplay(null);
    if (!api || readonly || !caseData) return;
    api
      .ping()
      .then((value) => {
        if (!disposed)
          setSupported(
            (value as { operations?: string[] }).operations?.includes(
              "trainPatient",
            ) ?? false,
          );
      })
      .catch(() => {
        if (!disposed) setSupported(false);
      });
    api
      .listRuns?.({ caseHash: caseData.caseHash })
      .then((value) => {
        if (!disposed && value.caseHash === currentHash.current)
          setRuns(value.runs);
      })
      .catch(() => {});
    return () => {
      disposed = true;
    };
  }, [api, caseData?.caseHash]);
  useEffect(() => {
    if (!api) return;
    return api.onEvent((event: BridgeEvent) => {
      if (event.op !== "trainPatient") return;
      const progress = event.progress as typeof event.progress & {
        metrics?: TrainingStats;
      };
      if (progress?.metrics) setLive(progress.metrics);
      if (event.event === "cancelled")
        setNote("Cancelled. Any available checkpoint stays on this Mac.");
    });
  }, [api]);
  async function refresh() {
    if (!api || !caseData) return;
    const result = await api.listRuns({ caseHash: caseData.caseHash });
    if (result.caseHash === currentHash.current) setRuns(result.runs);
  }
  async function train(resume?: TrainingRun) {
    if (!api || !caseData) return;
    setWorking(true);
    if (resume) {
      setBudget(resume.config.budgetSeconds);
      setSeed(resume.config.seed);
    }
    setLive(null);
    setRun(null);
    setReplay(null);
    onReplay(null);
    setNote("Preparing source-grid simulation…");
    try {
      const result = await api.trainPatient(
        resume
          ? { caseHash: caseData.caseHash, resumeRunId: resume.runId }
          : {
              caseHash: caseData.caseHash,
              budgetSeconds: budget,
              seed,
              routeId: route?.route_id,
            },
      );
      if (result.caseHash !== currentHash.current)
        throw new Error(
          "Training result belongs to another case and was withheld.",
        );
      setRun(result);
      setNote(
        result.status === "cancelled"
          ? "Cancelled checkpoint retained locally."
          : "Run completed. Inspect the actual update counts and independent replay status.",
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (/operation cancelled/i.test(message))
        setNote("Cancelled. Any available checkpoint stays on this Mac.");
      else onError(message);
    } finally {
      setWorking(false);
      void refresh().catch((error) => onError(String(error)));
    }
  }
  async function showReplay(selectedRun: TrainingRun, requestedStep?: number) {
    if (!api || !caseData) return;
    const generation = ++replayGeneration.current;
    setWorking(true);
    try {
      const result = await api.replayTraining({
        caseHash: caseData.caseHash,
        runId: selectedRun.runId,
        step: requestedStep,
      });
      const checked = await hydrateCertifiedReplay(result, caseData, api);
      if (
        generation !== replayGeneration.current ||
        result.caseHash !== currentHash.current
      )
        return;
      setRun({ ...selectedRun, training: result.training });
      setReplay(checked);
      setStep(result.step);
      onReplay(checked);
    } catch (error) {
      setReplay(null);
      onReplay(null);
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      if (generation === replayGeneration.current) setWorking(false);
    }
  }
  return (
    <div className="native-refinement">
      <div className="planning-title">
        <span className="eyebrow">PATIENT-SPECIFIC OPTIMIZATION</span>
        <h2>Refine within this case</h2>
        <p>
          Train a scratch policy in a frozen source-grid simulation. Search
          alternatives remain available.
        </p>
      </div>
      <div className="simulation-contract">
        <FlaskConical size={15} />
        <div>
          <strong>Discrete native-cell research model</strong>
          <span>Source-cell brush and complete-tool geometry.</span>
        </div>
      </div>
      <details className="refinement-contract">
        <summary>
          Frozen run assumptions <ChevronDown size={13} />
        </summary>
        <p>
          Patient image, supplied anatomy, access window, tool geometry, reward
          and optimization/selection worlds are fixed for a run. Final worlds
          are kept separate.
        </p>
        <p>
          Source cells are removed only when fully inside the declared brush.
          Partial contact stays retained.
        </p>
      </details>
      <div className="refinement-fields">
        <label>
          Learning budget
          <select
            value={budget}
            onChange={(event) => setBudget(Number(event.target.value))}
            disabled={blocked}
          >
            <option value={15}>15 seconds</option>
            <option value={30}>30 seconds</option>
            <option value={60}>60 seconds</option>
            <option value={120}>120 seconds</option>
          </select>
        </label>
        <label>
          Seed
          <input
            type="number"
            min={0}
            max={2147483647}
            value={seed}
            onChange={(event) => setSeed(Number(event.target.value))}
            disabled={blocked}
          />
        </label>
      </div>
      <p className="refinement-budget-note">
        Preparation and independent checking add time beyond the learning
        budget.
      </p>
      <button
        className="primary-button"
        disabled={
          !supported ||
          readonly ||
          blocked ||
          !caseData ||
          !route?.geometry.feasible ||
          !Number.isInteger(seed) ||
          seed < 0 ||
          seed > 2147483647
        }
        onClick={() => void train()}
      >
        {working ? (
          <LoaderCircle className="spin" size={15} />
        ) : (
          <Sparkles size={15} />
        )}{" "}
        Freeze assumptions & train
      </button>
      {!route?.geometry.feasible && (
        <p className="refinement-hint">
          Choose a feasible route A to supply a hypothetical access window and
          instrument.
        </p>
      )}
      {readonly && (
        <div className="capability-note">
          Training runs in the Mac app. This public-case preview is read-only.
        </div>
      )}
      {note && (
        <p className="training-status" role="status">
          {note}
        </p>
      )}
      {stats && (
        <>
          <div className="training-metrics">
            <div>
              <strong>{stats.gradient_steps ?? 0}</strong>
              <span>Gradient updates</span>
            </div>
            <div>
              <strong>{stats.optimization_environment_steps ?? 0}</strong>
              <span>Optimization steps</span>
            </div>
            <div>
              <strong>{stats.selection_environment_steps ?? 0}</strong>
              <span>Selection steps</span>
            </div>
          </div>
          <div className="parameter-change">
            Actor parameters{" "}
            <span>
              {stats.actor_parameters_changed === true
                ? "Updated"
                : stats.actor_parameters_changed === false
                  ? "Unchanged"
                  : "Awaiting final record"}
            </span>
          </div>
          {stats.selected_selection_return != null && (
            <div className="parameter-change">
              Selected surrogate return{" "}
              <span>
                {stats.selected_selection_return.toFixed(2)}{" "}
                <small>
                  (initial{" "}
                  {stats.initial_selection_return?.toFixed(2) ?? "unavailable"})
                </small>
              </span>
            </div>
          )}
          <SelectionCurve points={stats.selection_history ?? []} />
          <div className={`replay-gate ${accepted ? "accepted" : ""}`}>
            <strong>
              {accepted
                ? stats.replay?.stepCount === 0
                  ? "STOP selected · no modeled removal"
                  : "Independent native geometry accepted"
                : "Removal replay withheld"}
            </strong>
            <p>
              {accepted
                ? stats.replay?.stepCount === 0
                  ? "The selected policy stopped without a removal stroke. The accepted zero-action sequence is retained as the actual result."
                  : "Declared source-grid removal only. Missing functional and vascular evidence remains unassessed."
                : (stats.replay_status?.replace(/_/g, " ") ??
                  "Training and independent checking have not completed.")}
            </p>
          </div>
        </>
      )}
      {accepted && run && (
        <div className="replay-controls">
          <button
            className="outline-button"
            disabled={blocked}
            onClick={() => void showReplay(run)}
          >
            <Play size={13} /> Inspect selection replay
          </button>
          {replay && (
            <>
              <label htmlFor="replay-step">
                Modeled step{" "}
                <span>
                  {step} / {replay.result.stepCount}
                </span>
              </label>
              <input
                id="replay-step"
                type="range"
                min={0}
                max={replay.result.stepCount}
                value={step}
                onChange={(event) => setStep(Number(event.target.value))}
                onPointerUp={(event) =>
                  void showReplay(run, Number(event.currentTarget.value))
                }
                onKeyUp={(event) =>
                  void showReplay(run, Number(event.currentTarget.value))
                }
                disabled={blocked}
              />
              <dl>
                <dt>Removed target</dt>
                <dd>
                  {replay.result.simulatedRemovedTargetVolumeMm3.toFixed(1)} mm³
                </dd>
                <dt>Removed normal</dt>
                <dd>
                  {replay.result.simulatedRemovedNormalVolumeMm3.toFixed(1)} mm³
                </dd>
                <dt>Residual target</dt>
                <dd>
                  {replay.result.modeledResidualTargetVolumeMm3.toFixed(1)} mm³
                </dd>
              </dl>
              <button
                className="text-button"
                onClick={() => {
                  replayGeneration.current++;
                  setReplay(null);
                  onReplay(null);
                }}
              >
                Return to source annotations
              </button>
            </>
          )}
          <button
            className="text-button"
            disabled={blocked}
            onClick={() =>
              void api
                ?.exportCandidate({ caseHash: run.caseHash, runId: run.runId })
                .then((result) => {
                  if (result?.exported)
                    setNote(
                      "Independently checked candidate exported locally.",
                    );
                })
                .catch((error) => onError(String(error)))
            }
          >
            <ArrowDownToLine size={12} /> Export checked candidate
          </button>
        </div>
      )}
      {runs.length > 0 && (
        <details className="saved-runs">
          <summary>
            Local run history <span>{runs.length}</span>
          </summary>
          {runs.map((item) => (
            <div className="saved-run" key={item.runId}>
              <div>
                <strong>
                  {item.createdAt
                    ? new Date(item.createdAt * 1000).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : "Local run"}
                </strong>
                <span>
                  {item.status.replace(/_/g, " ")} · seed {item.config.seed}
                </span>
              </div>
              {item.status === "cancelled" && item.hasCheckpoint && (
                <button
                  className="icon-button"
                  aria-label="Resume cancelled run"
                  disabled={blocked}
                  onClick={() => void train(item)}
                >
                  <RotateCcw size={14} />
                </button>
              )}
              {item.hasAcceptedReplay && (
                <button
                  className="icon-button"
                  aria-label="Recheck and inspect saved replay"
                  disabled={blocked}
                  onClick={() => void showReplay(item)}
                >
                  <Play size={14} />
                </button>
              )}
            </div>
          ))}
        </details>
      )}
    </div>
  );
}
