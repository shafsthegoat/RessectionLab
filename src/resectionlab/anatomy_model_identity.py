"""Preserve model identity across audited non-inference source changes."""

from hashlib import file_digest, sha256
from pathlib import Path


LEGACY_POC_LAUNCHER_SHA = "081620aca0fd0f4fe3b9d9ceda541686743392d454b6beb5134c00e45de83e0a"
AUDITED_LAUNCHER_SHA = "065caba2b0e2073b660f346913cd1bb073f9a5c00e0e439b92f1a9418156a756"
AUDITED_SUPERVISOR_SHA = "b671b9ae08a1ee1f29051a2f60fa4ae69684b7d68ea3551393b4d3d220eb2003"


def _module_sha(name):
    with Path(__file__).with_name(name).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def poc_worker_identity(launcher_sha256):
    if launcher_sha256 == LEGACY_POC_LAUNCHER_SHA:
        return launcher_sha256
    supervisor = _module_sha("poc_worker_supervision.py")
    # Worker and preprocessing are unchanged. The Darwin CPU SynthStrip
    # guard adds its own cache identity field; other executions retain theirs.
    # Any later change to either source must use a distinct cache identity.
    if launcher_sha256 == AUDITED_LAUNCHER_SHA and supervisor == AUDITED_SUPERVISOR_SHA:
        return LEGACY_POC_LAUNCHER_SHA
    return sha256((launcher_sha256 + supervisor).encode("ascii")).hexdigest()


# The model runtime is identical across this admission-only pair. Crop metadata
# is backfilled separately, with its own key and without changing inferred arrays.
LEGACY_ADAPTER_SHA = "580eb66da25c2001c2960bdc1606ed9b2e1fc9a2472eac1112f1c3005f1f1cd8"
AUDITED_CROP_ADAPTER_SHA = "ea7076f62cf8c47dabb187e628c065c6f009a58a69bcc51721b5eb263c23c032"
AUDITED_CROP_HELPER_SHA = "2d8cd7f8c64bfafe2822f0eb7cbbd07fede6e1f316bb93ca43dd80f123d55901"


# The SIAM dispatch and multi-series scheduler leave inference for previously
# eligible jobs unchanged. Source series, model options and weights are still
# hashed separately. New SIAM runs retain both their adapter and worker hashes.
SIAM_DISPATCH_SHA = "6593e2cca605d832298118eeda38a3db89f06e1c7a9bc4218d9f6321a4d0246b"
MULTISERIES_REGISTRY_SHA = "66756f751cb8f85cc2f61664c971ac28f27349959c3519a66b2009eb8e140d7b"
PREVIOUS_REGISTRY_SHA = "c3e2d789fbc383e2e41f92ad26cf62a8045283794073ee156a6473b12ae2cbc6"


# Adding the opt-in four-channel stage leaves the existing inference paths unchanged.
BRATS_REGISTRY_SHA = "228ca8ba264f2824c6f771121961df07cc232b895260a4aafeb1fc8fe7e1322a"


def preparation_asset_identities(files):
    identities = dict(files)
    if "brats_worker" not in files and files.get("registry") == BRATS_REGISTRY_SHA:
        identities["registry"] = MULTISERIES_REGISTRY_SHA
    if "poc_worker" in files:
        identities["poc_worker"] = poc_worker_identity(files["poc_worker"])
    if "siam_worker" not in files:
        if identities.get("registry") == MULTISERIES_REGISTRY_SHA:
            identities["registry"] = PREVIOUS_REGISTRY_SHA
        if files.get("adapter") == SIAM_DISPATCH_SHA:
            identities["adapter"] = AUDITED_CROP_ADAPTER_SHA
    if identities.get("adapter") == AUDITED_CROP_ADAPTER_SHA:
        crop = _module_sha("anatomy_tumor_coverage.py")
        identities["adapter"] = (LEGACY_ADAPTER_SHA if crop == AUDITED_CROP_HELPER_SHA
                                 else sha256((AUDITED_CROP_ADAPTER_SHA + crop).encode("ascii")).hexdigest())
    return identities
