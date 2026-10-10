"""Candidate tracked entry: launchers/hbe_v5_ordinal10_continuation_v1.py.

No release exists. This source declares only the frozen tension N24 S60 row.
"""
SPEC = {
    "index": 10,
    "run_id": "tension:N24:S60:reference",
    "adapter_source": "launchers/hbe_v5_ordinal10_continuation_v1.py",
    "sidecar_directory": ("outputs/mechanics/hbe-v5-later-continuation-v1/"
                          "10-tension-N24-S60-reference/attempt-01"),
    "native_output_directory": ("outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
                                "10-tension-N24-S60-reference/attempt-01"),
    "expected_preflight_hint_opens": 18,
    "expected_preflight_hint_bytes": 2_908_379_069,
}
if __name__ == "__main__":
    raise SystemExit("Run the exact reviewed runtime source with --ordinal 10")
