# Sealed planning comparison: source integration

The annotation evaluator now checks all SEARCH/IL/RL plan seals and accepted full-tool histories before loading a private reference. It reuses the existing independent streaming geometry calculation and reports annotated contact, unknown coverage and complete-tool encounters. It does not infer removal or injury.

Ten focused generated controls pass (0.88 seconds). Independent producer/consumer inspection found two real interface failures: nominal versus committed histories carry different outcome-scope labels, and STOP rows carry neither label. Both are repaired with explicit exact replay-history binding; all physical fields must remain equal. A 0.528-second generated producer seam check passes motion-only, STOP-only and motion-then-STOP, and rejects changed replay before reference access. The original failure is retained. No patient evaluation or learning was run.

The caller must supply a qualified reference and the exact source-bound bundle hash, and must verify successful worker and parent termination before calling this evaluator. A surviving sealed file after a late worker failure is not sufficient. This adapter does not provide process isolation or perform anatomical registration; actual patient scoring remains under the experiment's resource limits and intended-use QC.
