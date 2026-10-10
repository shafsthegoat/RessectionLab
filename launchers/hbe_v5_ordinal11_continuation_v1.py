"""Candidate tracked entry: launchers/hbe_v5_ordinal11_continuation_v1.py.

The row-10 sidecar does not yet exist. A future separately reviewed release
must bind its exact native receipt, sidecar, releases, source commit and node/
element byte sizes. Without that extension descriptor this entry refuses.
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
    "expected_preflight_hint_bytes": None,
}
if __name__ == "__main__":
    raise SystemExit("Run the exact reviewed runtime source with --ordinal 11")
