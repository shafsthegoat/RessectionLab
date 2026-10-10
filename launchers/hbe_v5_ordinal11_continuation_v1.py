"""Candidate tracked entry: launchers/hbe_v5_ordinal11_continuation_v1.py.

The actual row-10 sidecar and independently replayed numerical result are
retained. The source-bound runtime must still authenticate their exact native
receipt, sidecar, releases and source commit before any row-11 native call.
"""
SPEC = {
    "index": 11,
    "run_id": "tension:N24:S120:reference",
    "adapter_source": "launchers/hbe_v5_ordinal11_continuation_v1.py",
    "sidecar_directory": ("outputs/mechanics/hbe-v5-later-continuation-v1/"
                          "11-tension-N24-S120-reference/attempt-01"),
    "native_output_directory": ("outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
                                "11-tension-N24-S120-reference/attempt-01"),
    "expected_preflight_hint_opens": 20,
    # Frozen rows 0..9: 2,908,379,069 B. Actual row-10 native receipt:
    # nodes.log 118,531,657 B and elements.log 74,276,117 B.
    "expected_preflight_hint_bytes": 3_101_186_843,
}
if __name__ == "__main__":
    raise SystemExit("Run the exact reviewed runtime source with --ordinal 11")
