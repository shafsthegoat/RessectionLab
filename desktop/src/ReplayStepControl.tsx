/** Requested thumb position is separate from the last independently checked display. */
export function ReplayStepControl({
  id = "replay-step",
  requestedStep,
  appliedStep,
  stepCount,
  disabled,
  onRequest,
}: {
  id?: string;
  requestedStep: number;
  appliedStep: number;
  stepCount: number;
  disabled: boolean;
  onRequest: (step: number) => void;
}) {
  const pending = requestedStep !== appliedStep;
  return (
    <div className="replay-step-control">
      <label htmlFor={id}>
        Showing modeled step{" "}
        <span>
          {appliedStep} / {stepCount}
        </span>
      </label>
      <input
        id={id}
        type="range"
        min={0}
        max={stepCount}
        value={requestedStep}
        aria-valuetext={
          pending
            ? `Requested step ${requestedStep}; showing checked step ${appliedStep} of ${stepCount}`
            : `Showing checked step ${appliedStep} of ${stepCount}`
        }
        aria-describedby={pending ? `${id}-pending` : undefined}
        onChange={(event) => onRequest(Number(event.target.value))}
        disabled={disabled || stepCount === 0}
      />
      {pending && (
        <p
          className="replay-step-pending"
          id={`${id}-pending`}
          role="status"
        >
          Updating to step {requestedStep}… Quantities and overlay still show
          step {appliedStep}.
        </p>
      )}
    </div>
  );
}
