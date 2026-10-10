"""One exact authorized public SEG; selected native planes and raw-bit checks."""
import hashlib
import io
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "data/acquisition/remind-select-public-v1/verified/ReMIND-037/88911f75-9082-427c-8cd6-9784aecf46c0/18b4c53d-6ec3-4ad5-b6b2-3fe1292e4b91.dcm"
SOURCE_SHA = "f8ad2b9d4900b8d9e568ce09d63d9949bb3b838a039a5d17111e14a175414aee"
SOURCE_SIZE = 8741376


def save(path, value):
    with path.open("x") as stream: json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


def main():
    started = time.monotonic(); output = HERE / "exact037-seg-diagnostic-v1"; output.mkdir(exist_ok=False)
    report = {"status": "failed", "source_bytes_read": 0, "source_sha256": SOURCE_SHA,
              "private_reference_loaded": False, "MRI_loaded": False, "anatomy_pass": False}
    def stop(*args): raise TimeoutError("50-second public SEG diagnostic cap")
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop); signal.alarm(50)
    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = Path(args[0]).resolve()
            if path.is_relative_to(ROOT / "data") and path != SOURCE:
                raise PermissionError("Only the one authorized SEG payload may be opened")
            if path.suffix == ".npy" and path.name != "cerebrum_source_label.npy":
                raise PermissionError("Only saved public support array is authorized")
        if event in ("socket.connect", "socket.bind", "subprocess.Popen", "os.system"):
            raise PermissionError("No external work")
    sys.addaudithook(guard)
    try:
        before = SOURCE.stat(); assert before.st_size == SOURCE_SIZE
        with SOURCE.open("rb", buffering=0) as stream: raw = stream.read(SOURCE_SIZE)
        report["source_bytes_read"] = len(raw)
        after = SOURCE.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        assert len(raw) == SOURCE_SIZE and hashlib.sha256(raw).hexdigest() == SOURCE_SHA
        ds = pydicom.dcmread(io.BytesIO(raw), force=False); del raw
        assert str(ds.PatientID) == "ReMIND-037" and str(ds.Modality) == "SEG"
        assert str(ds.SeriesInstanceUID) == "1.3.6.1.4.1.14519.5.2.1.38908981817422885406628241891552387395"
        assert str(ds.SegmentationType) == "BINARY" and int(ds.BitsAllocated) == int(ds.BitsStored) == 1
        assert int(ds.HighBit) == int(ds.PixelRepresentation) == 0 and type(ds.PixelData) is bytes
        rows, columns, frames = int(ds.Rows), int(ds.Columns), int(ds.NumberOfFrames)
        assert (columns, rows, frames) == (399, 513, 339)
        expected = (rows*columns*frames+7)//8
        assert len(ds.PixelData) in (expected, expected+1)
        packed = np.frombuffer(ds.PixelData, np.uint8)
        def samples(x, y, z):
            indices = (z*rows+y)*columns+x
            return ((packed[indices//8] >> (indices % 8)) & 1).astype(np.uint8)
        folder = HERE / "actual-public-qc-v1/ReMIND-037-crop-mr"
        result_raw = (folder / "conversion-result.json").read_bytes()
        assert hashlib.sha256(result_raw).hexdigest() == "e3671d0089279f786225f63180159ae656f37d086aa159c5f622808a3198d41f"
        result = json.loads(result_raw)
        header_raw = (HERE / "actual-public-qc-v1/ReMIND-037-headers/result.json").read_bytes()
        assert hashlib.sha256(header_raw).hexdigest() == result["header_snapshot_sha256"]
        header = json.loads(header_raw); source_header = next(r for r in header["series"] if r["kind"] == "cerebrum")
        order = np.asarray(source_header["geometry"]["sorted_source_indices"], int)
        assert order.tolist() == list(range(frames))
        path = folder / "cerebrum_source_label.npy"
        raw = path.read_bytes(); assert hashlib.sha256(raw).hexdigest() == result["artifacts"][path.name]["sha256"]; del raw
        saved = np.load(path, mmap_mode="r", allow_pickle=False)
        transform = np.asarray(result["annotations"]["cerebrum"]["placement"]["target_index_to_source_index"])
        x, y = np.indices(saved.shape[:2], dtype=float); matches = 0
        for z in range(saved.shape[2]):
            xyz = transform[:3, 0, None, None]*x+transform[:3, 1, None, None]*y+(transform[:3, 2]*z+transform[:3, 3])[:, None, None]
            nearest = np.floor(xyz+.5).astype(np.int64)
            raw_plane = samples(nearest[0], nearest[1], order[nearest[2]])
            assert np.array_equal(raw_plane, saved[:, :, z]); matches += raw_plane.size
        # Fresh pydicom selected-frame decoding is checked against an independent
        # bit-address oracle. No full native stack allocation is needed.
        gx, gy = np.meshgrid(np.arange(columns), np.arange(rows), indexing="xy")
        selected = (0, 1, 2, 149, 150, 151, 152, 153, 168, 169, 170, 336, 337, 338)
        plane_records = []
        raw_cache = {}
        for z in selected:
            oracle = samples(gx, gy, z)
            ds.pixel_array_options(index=z, raw=True, decoding_plugin="", use_v2_backend=False)
            decoded = ds.pixel_array
            assert np.array_equal(decoded, oracle)
            plane_records.append({"frame_index": z, "positive_voxels": int(oracle.sum()),
                "raw_and_pydicom_equal": True, "raw_plane_sha256": hashlib.sha256(oracle.tobytes()).hexdigest()})
            raw_cache[z] = oracle
        adjacent = []
        for z in selected:
            if z+1 in raw_cache:
                a, b = raw_cache[z], raw_cache[z+1]
                union = int(np.count_nonzero(a | b)); different = int(np.count_nonzero(a != b))
                adjacent.append({"frames": [z, z+1], "difference_fraction_of_union": different/union if union else None})
        # Orthogonal source-label planes at the corresponding saved-QC points.
        # These are encoded label samples only, with no MRI/background image.
        from resectionlab.remind_planning_qc import png
        source_point = np.floor((transform @ [24, 73, 50, 1])[:3]+.5).astype(int)
        shape = (columns, rows, frames); views = []
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            coordinates = np.indices(tuple(shape[a] for a in others), dtype=np.int64)
            xyz = [None]*3; xyz[axis] = source_point[axis]
            for i, other in enumerate(others): xyz[other] = coordinates[i]
            plane = samples(*xyz).T
            name = f"encoded-source-axis{axis}-index{source_point[axis]}.png"
            png(output/name, np.repeat((plane*255)[:, :, None], 3, axis=2))
            views.append({"file": name, "axis": axis, "source_index": int(source_point[axis]),
                "sha256": hashlib.sha256((output/name).read_bytes()).hexdigest(), "positive_is_white": True})
        report.update(status="encoded_source_pattern_diagnostic_complete_anatomy_unreviewed",
            shape_xyz=shape, source_transfer_syntax=str(ds.file_meta.TransferSyntaxUID),
            pixel_data_bytes=len(ds.PixelData), expected_contiguous_bit_bytes=expected,
            samples_per_frame=rows*columns, bits_per_frame_mod8=(rows*columns)%8,
            frame_padding_policy="continuous DICOM bitstream, no per-frame byte padding assumed",
            selected_native_planes=plane_records, selected_adjacent_frame_disagreement=adjacent,
            saved_crop_voxels_checked_against_original_bit_addresses=matches, saved_crop_all_raw_samples_equal=True,
            source_frame_order_matches_saved=True, encoded_source_views=views,
            source_algorithm_type=str(ds.SegmentSequence[0].SegmentAlgorithmType),
            source_algorithm_name=str(getattr(ds.SegmentSequence[0], "SegmentAlgorithmName", "")),
            conclusion="Saved NN crop is exactly selected source bits; source-native label views separate encoded pattern from display/MRI overlay. Cause and anatomical validity remain unreviewed.")
    except BaseException as error: report.update(error=type(error).__name__+":"+str(error))
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            executing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save(output / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "error", "source_bytes_read", "elapsed_seconds", "peak_rss_bytes")}))
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
