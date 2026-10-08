"""Pinned VitalDB numeric component import and retrospective observed replay.

Packet layout derived from vitaldb 1.7.2 utils.py, MIT, Copyright (c) 2021 VitalDB.
The complete upstream notice is retained in
artifacts/vitaldb-recorded-component-v1/upstream-MIT-LICENSE.txt.
Archive SHA256: 70e0ce784b13d52bbf6a315b742ab06d5ed0599b8b01f76182971a391dd64130
utils.py SHA256: b0e88b1365c8d9a88814123c9a5b5a5d3c696a5c4849c7db3ebda476cc7134e3

Only the authenticated local PhysioNet 1.0.0 case3 file is admitted. No waveform
or unrelated numeric values are decoded. This module has no network dependency.
Whole-case QC is a separate, explicitly retrospective interface. Replay frames
release only records whose original recording timestamps are <= the clock.
The API is an evidence boundary, not a Python sandbox against private access.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import struct
from typing import BinaryIO
import zlib

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "manifests/vitaldb-development-v1.json"
DECLARATION_SHA256 = "8f2053f5b3a435cc3b84c27e868c22be93dd1e6995bd44ebc414cc9c5e5c98a0"
COMPONENT_DECLARATION = ROOT / "manifests/vitaldb-native-component-v1.json"
COMPONENT_DECLARATION_SHA256 = "7170c6d5c521b845e937eb69b16d0083650823ae8da4a6da201eeff71bfe9679"
CLOCK_REVIEW = ROOT / "manifests/vitaldb-native-clock-v1.json"
CLOCK_REVIEW_SHA256 = "11bf6b1b97cdf894a64f4c0110a2d52f053bcb6ca52237bf2515c417df253dc8"
SOURCE_SHA256 = "573db0941d580167833f84497f5d1c4a1391443eeec3f95852287ea09db024e4"
SOURCE_BYTES = 6537712
TRACKS = {
    "Orchestra/PPF20_RATE": "mL/h", "Orchestra/PPF20_VOL": "mL",
    "Orchestra/RFTN20_RATE": "mL/h", "Orchestra/RFTN20_VOL": "mL",
    "Solar8000/NIBP_MBP": "mmHg", "Solar8000/NIBP_SBP": "mmHg",
    "Solar8000/NIBP_DBP": "mmHg",
}
MISSING_ARTERIAL_TRACKS = ("Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP")
FORMATS = {1: "f", 2: "d", 3: "b", 4: "B", 5: "h", 6: "H", 7: "l", 8: "L"}
LIMITS = {"decompressed_bytes": 134217728, "packet_bytes": 4194304,
          "packets": 1000000, "records": 250000, "header_bytes": 4096,
          "tracks": 1024, "devices": 256}
_IMPORTED_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LIMITATIONS = (
    "Retrospective observed component replay; DEVELOPMENT case3/subject2861, not glioma or RL experience.",
    "Clock uses unmapped native source seconds; acquisition/display latency and inter-device synchronization error are unavailable.",
    "Native timestamps have an unverified epoch near 2100; recording-start normalization is unresolved. Raw native seconds are replayed unchanged.",
    "Source pump names may have been assigned retrospectively using pump data or anesthesia records; contemporaneous identity availability is unproved.",
    "A retained last observation is not a current measurement or continuous physiology; gaps remain missing.",
    "Pressure channels are intermittent noninvasive cuff observations (NIBP). Required arterial channels are absent; the original arterial request failed.",
    "Monitor records may repeat the previous cuff result; packet publication time and source-record age do not establish cuff acquisition time, measurement freshness, or cuff-cycle count.",
    "Pump rates/volumes do not establish circulation delivery, bolus timing, causal drug effects, or eligible actions/rewards.",
    "Source release removed all-zero and fewer-than-ten-sample tracks; import completeness is only against the released file.",
)


class VitalDBError(ValueError):
    """Refusal, unsupported format, or integrity failure; never repaired silently."""


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hash() -> str:
    if _sha(Path(__file__).read_bytes()) != _IMPORTED_CODE_SHA256:
        raise VitalDBError("Reader source changed since import; restart before admission")
    return _IMPORTED_CODE_SHA256


def _declaration() -> dict:
    data = DECLARATION.read_bytes()
    if _sha(data) != DECLARATION_SHA256:
        raise VitalDBError("Prospective role/source declaration changed")
    companion = COMPONENT_DECLARATION.read_bytes()
    if _sha(companion) != COMPONENT_DECLARATION_SHA256:
        raise VitalDBError("Prospective native-component declaration changed")
    evidence = json.loads(companion)["native_header_evidence"]
    if _sha((ROOT / evidence["path"]).read_bytes()) != evidence["sha256"]:
        raise VitalDBError("Prospective native-header evidence changed")
    if _sha(CLOCK_REVIEW.read_bytes()) != CLOCK_REVIEW_SHA256:
        raise VitalDBError("Native clock review changed")
    return json.loads(data)


def _admission_evidence() -> dict:
    files = {
        "rights_text": ("artifacts/vitaldb-recorded-component-v1/PhysioNet-CC-BY-4.0.txt", "9a78e7f22742dde9f66ae235ec793ba2212019dc4d0ced75c4da09ced0b35fb2"),
        "mit_notice": ("artifacts/vitaldb-recorded-component-v1/upstream-MIT-LICENSE.txt", "87fa7cab278f36431b02007807f76c1b0b4d8534777d2337f1f1902294eea1d2"),
        "rights_and_parser_receipt": ("artifacts/vitaldb-recorded-component-v1/rights-and-parser.json", "eb43776083dced77505dee6aa57c3f9bce4662cf34c140056f3eff67568913a5"),
        "parser_archive": ("build/vitaldb-source-v1/vitaldb-1.7.2.tar.gz", "70e0ce784b13d52bbf6a315b742ab06d5ed0599b8b01f76182971a391dd64130"),
        "upstream_utils": ("build/vitaldb-source-v1/upstream-utils.py", "b0e88b1365c8d9a88814123c9a5b5a5d3c696a5c4849c7db3ebda476cc7134e3"),
    }
    result = {}
    for name, (relative, expected) in files.items():
        with (ROOT / relative).open("rb") as handle:
            # Legal/source reference material has a separate 1 MiB ceiling.
            data = handle.read(1048577)
        if len(data) > 1048576 or _sha(data) != expected:
            raise VitalDBError(f"Rights/parser admission evidence changed: {name}")
        result[name + "_sha256"] = expected
    return result


class _GzipStream:
    """Bounded single-member gzip reader; validates CRC, ISIZE and exact EOF.

    zlib handles trailer validation. Every compressed byte is hashed during the
    actual parse too, binding parsed content despite a concurrent file mutation.
    """
    def __init__(self, stream: BinaryIO, limit: int = LIMITS["decompressed_bytes"]):
        self.stream = stream
        self.limit = limit
        self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.buffer = bytearray()
        self.pending = b""
        self.total = 0
        self.compressed = 0
        self.digest = hashlib.sha256()
        self.done = False

    def read(self, n: int) -> bytes:
        if n < 0 or n > LIMITS["packet_bytes"]:
            raise VitalDBError("Unbounded read refused")
        while len(self.buffer) < n and not self.done:
            if self.decoder.eof:
                if self.decoder.unused_data or self.pending or self.stream.read(1):
                    raise VitalDBError("Trailing bytes or concatenated gzip members refused")
                self.done = True
                break
            data = self.pending
            if not data:
                data = self.stream.read(min(65536, SOURCE_BYTES - self.compressed + 1))
                if not data:
                    raise VitalDBError("Incomplete gzip member/CRC/EOF")
                self.compressed += len(data)
                self.digest.update(data)
                if self.compressed > SOURCE_BYTES:
                    raise VitalDBError("Compressed byte bound exceeded")
            try:
                output = self.decoder.decompress(data, min(1048576, self.limit - self.total + 1))
            except zlib.error as exc:
                raise VitalDBError(f"Invalid gzip member/CRC: {exc}") from exc
            self.pending = self.decoder.unconsumed_tail
            self.total += len(output)
            if self.total > self.limit:
                raise VitalDBError("Decompressed byte bound exceeded")
            self.buffer.extend(output)
        result = bytes(self.buffer[:n])
        del self.buffer[:n]
        return result

    def exact(self, n: int) -> bytes:
        data = self.read(n)
        if len(data) != n:
            raise VitalDBError("Truncated header or packet")
        return data

    def skip(self, n: int) -> None:
        while n:
            size = min(n, 65536)
            self.exact(size)
            n -= size


class _Cursor:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def take(self, n: int) -> bytes:
        if n < 0 or n > self.remaining:
            raise VitalDBError("Field exceeds packet length")
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result

    def number(self, code: str):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def string(self) -> str:
        size = self.number("L")
        if size > LIMITS["header_bytes"]:
            raise VitalDBError("Metadata string bound exceeded")
        try:
            return self.take(size).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise VitalDBError("Invalid metadata UTF-8") from exc

    def finish(self) -> None:
        if self.remaining:
            raise VitalDBError("Unsupported trailing packet fields")


@dataclass(frozen=True)
class NumericObservation:
    track: str
    unit: str
    source_timestamp_seconds: float
    timestamp_bytes_hex: str
    record_header_extension_hex: str
    packet_ordinal: int
    track_ordinal: int
    value: float | int | None
    value_kind: str
    value_bytes_hex: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Recording:
    observations: tuple[NumericObservation, ...]  # Original packet order, immutable.
    header_json: bytes
    track_headers_json: bytes
    parse_stats_json: bytes
    binding_json: bytes


def _validate_track(full_name: str, kind: int, fmt: int, unit: str,
                    sample_rate: float, gain: float, offset: float) -> None:
    if kind != 2 or fmt != 1 or unit != TRACKS[full_name]:
        raise VitalDBError(f"Wrong selected track type/format/unit: {full_name}")
    if not all(math.isfinite(v) for v in (sample_rate, gain, offset)):
        raise VitalDBError("Nonfinite selected track calibration")
    # This narrow numeric reader preserves upstream TYPE_NUM values without a
    # waveform gain transform. Unexpected calibration is unsupported, not repaired.
    if sample_rate != 0 or gain != 1 or offset != 0:
        raise VitalDBError(f"Unsupported numeric calibration: {full_name}")


def _read_local(path: str | Path, *, header_only: bool = False) -> _Recording:
    declaration = _declaration()
    admission_evidence = _admission_evidence()
    if not isinstance(path, (str, Path)) or "://" in str(path):
        raise VitalDBError("Only a local file path is allowed")
    path = Path(path)
    with path.open("rb") as raw:
        info = os.fstat(raw.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != SOURCE_BYTES:
            raise VitalDBError("Missing file or frozen compressed size mismatch")
        hasher, hashed = hashlib.sha256(), 0
        while hashed <= SOURCE_BYTES:
            chunk = raw.read(min(65536, SOURCE_BYTES + 1 - hashed))
            if not chunk:
                break
            hashed += len(chunk)
            hasher.update(chunk)
        if hashed != SOURCE_BYTES or hasher.hexdigest() != SOURCE_SHA256:
            raise VitalDBError("Frozen source SHA256 mismatch")
        raw.seek(0)
        stream = _GzipStream(raw)
        if stream.exact(4) != b"VITA":
            raise VitalDBError("Unsupported native file signature")
        version = struct.unpack("<L", stream.exact(4))[0]
        header_size = struct.unpack("<H", stream.exact(2))[0]
        if version not in (2, 3) or header_size not in (10, 26, 27):
            raise VitalDBError(f"Unsupported native version/header: {version}/{header_size}")
        h = _Cursor(stream.exact(header_size))
        header = {"version": version, "header_bytes": header_size,
                  "timezone_offset_minutes": h.number("h"), "instance_id": h.number("L"),
                  "program_version": h.number("L"), "source_dtstart": None, "source_dtend": None,
                  "packed": False}
        if h.remaining:
            header.update(source_dtstart=h.number("d"), source_dtend=h.number("d"))
            if not all(math.isfinite(header[key]) for key in ("source_dtstart", "source_dtend")):
                raise VitalDBError("Nonfinite file timestamp origin")
        if h.remaining:
            flag = h.number("B")
            if flag not in (0, 1):
                raise VitalDBError("Unsupported packed flag")
            header["packed"] = bool(flag)
        h.finish()
        # Actual native bytes contradict the release-relative identity assumption.
        # Preserve the full native clock. The possible 2100 epoch is not subtracted.
        header.update(clock_origin_seconds=None, clock_mapping="identity on native timestamps; no subtraction or resampling",
                      time_basis="unmapped_native_source_seconds", recording_start_mapping_status="unresolved",
                      clock_review_sha256=CLOCK_REVIEW_SHA256)
        devices, tracks, selected_headers = {}, {}, {}
        selected_counts: Counter = Counter()
        packet_counts: Counter = Counter()
        observations = []
        ordinal = max_packet = skipped_payload_bytes = 0
        while True:
            prefix = stream.read(5)
            if not prefix:
                break
            if len(prefix) != 5:
                raise VitalDBError("Truncated packet envelope")
            packet_type, length = struct.unpack("<BL", prefix)
            ordinal += 1
            if ordinal > LIMITS["packets"] or length > LIMITS["packet_bytes"]:
                raise VitalDBError("Packet count/size bound exceeded")
            max_packet = max(max_packet, length)
            packet_counts[packet_type] += 1
            if packet_type == 9:
                if length > LIMITS["header_bytes"]:
                    raise VitalDBError("Device header bound exceeded")
                p = _Cursor(stream.exact(length))
                did, device_type, name = p.number("L"), p.string(), p.string()
                if p.remaining:
                    p.string()  # Source port metadata is not replayed.
                p.finish()
                if did in devices or len(devices) >= LIMITS["devices"]:
                    raise VitalDBError("Duplicate device or device bound exceeded")
                devices[did] = name or device_type
            elif packet_type == 0:
                if length > LIMITS["header_bytes"]:
                    raise VitalDBError("Track header bound exceeded")
                p = _Cursor(stream.exact(length))
                tid, kind, fmt = p.number("H"), p.number("B"), p.number("B")
                name, unit = p.string(), p.string()
                min_display, max_display, color = p.number("f"), p.number("f"), p.number("L")
                sample_rate, gain, offset = p.number("f"), p.number("d"), p.number("d")
                monitor_type, did = p.number("B"), p.number("L")
                extension = {}
                for key, code in (("record_bytes", "L"), ("source_dtstart", "d"), ("source_dtend", "d")):
                    if p.remaining:
                        extension[key] = p.number(code)
                p.finish()
                if tid in tracks or len(tracks) >= LIMITS["tracks"]:
                    raise VitalDBError("Duplicate track ID or track bound exceeded")
                if did and did not in devices:
                    raise VitalDBError("Track refers to undeclared device")
                full_name = (devices[did] + "/" if did else "") + name
                tracks[tid] = full_name
                if full_name in TRACKS:
                    if full_name in selected_headers:
                        raise VitalDBError("Duplicate selected full track name")
                    _validate_track(full_name, kind, fmt, unit, sample_rate, gain, offset)
                    selected_headers[full_name] = {"track_id": tid, "name": full_name,
                        "type": kind, "format": fmt, "unit": unit, "sample_rate": sample_rate,
                        "gain": gain, "offset": offset, "monitor_type": monitor_type,
                        "device_id": did, "extension": extension}
            elif packet_type == 1:
                if length < 12:
                    raise VitalDBError("Short record header")
                info = stream.exact(12)
                info_size, tid = struct.unpack_from("<H", info, 0)[0], struct.unpack_from("<H", info, 10)[0]
                if info_size < 10 or info_size > LIMITS["header_bytes"] or 2 + info_size > length or tid not in tracks:
                    raise VitalDBError("Unsupported record header or unknown track")
                full_name = tracks[tid]
                if full_name not in TRACKS or header_only:
                    stream.skip(length - 12)
                    skipped_payload_bytes += length - 12
                    continue
                info_extension = stream.exact(info_size - 10)
                fmt = FORMATS[selected_headers[full_name]["format"]]
                size = struct.calcsize("<" + fmt)
                if length != 2 + info_size + size:
                    raise VitalDBError("Selected numeric record has wrong exact length")
                timestamp = struct.unpack_from("<d", info, 2)[0]
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise VitalDBError("Unreplayable nonfinite/negative source timestamp")
                value_bytes = stream.exact(size)
                value = struct.unpack("<" + fmt, value_bytes)[0]
                value_kind = "finite"
                if not math.isfinite(value):
                    value_kind = "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")
                    value = None
                selected_counts[full_name] += 1
                if len(observations) >= LIMITS["records"]:
                    raise VitalDBError("Selected record bound exceeded")
                observations.append(NumericObservation(full_name, TRACKS[full_name], timestamp,
                    info[2:10].hex(), info_extension.hex(), ordinal, selected_counts[full_name], value, value_kind, value_bytes.hex()))
            elif packet_type == 10:
                stream.skip(length)  # Raw device traffic is never interpreted.
                skipped_payload_bytes += length
            elif packet_type == 6:
                p = _Cursor(stream.exact(length))
                command = p.number("B")
                if command == 5:
                    count = p.number("H")
                    if count > LIMITS["tracks"]:
                        raise VitalDBError("Track-order command bound exceeded")
                    p.take(count * 2)
                elif command != 6:
                    raise VitalDBError("Unsupported native command")
                p.finish()  # Reset-events affects an excluded string track only.
            else:
                raise VitalDBError(f"Unsupported packet type {packet_type}")
        if not stream.done or stream.digest.hexdigest() != SOURCE_SHA256 or stream.compressed != SOURCE_BYTES:
            raise VitalDBError("Incomplete gzip validation or parsed bytes changed")
    if set(selected_headers) != set(TRACKS) or (not header_only and any(selected_counts[n] == 0 for n in TRACKS)):
        raise VitalDBError("Missing required numeric track/header/records")
    binding = {"source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
               "declaration_sha256": DECLARATION_SHA256, "reader_code_sha256": _code_hash(),
               "component_declaration_sha256": COMPONENT_DECLARATION_SHA256,
               "clock_review_sha256": CLOCK_REVIEW_SHA256, "admission_evidence": admission_evidence,
               "time_basis": "unmapped_native_source_seconds",
               "source_release": "PhysioNet VitalDB 1.0.0", "case_id": 3,
               "subject_id": 2861, "role": "DEVELOPMENT", "mode": "retrospective_observed_component_replay"}
    return _Recording(tuple(observations), _json(header), _json(selected_headers), _json({
        "packet_count": ordinal, "packet_type_counts": dict(packet_counts),
        "max_packet_bytes": max_packet, "compressed_bytes": stream.compressed,
        "decompressed_bytes": stream.total, "device_count": len(devices), "track_count": len(tracks),
        "skipped_payload_bytes": skipped_payload_bytes, "unrelated_values_decoded": 0,
        "complete_gzip_crc_eof_validated": True}), _json(binding))


def check_original_request(path: str | Path) -> dict:
    """Reproduce original arterial-spec failure using headers only, no values."""
    recording = _read_local(path, header_only=True)
    headers = json.loads(recording.track_headers_json)
    original = _declaration()["exact_tracks"]
    missing = sorted(set(original) - set(headers))
    units = {name: {"declared": unit, "actual_native": headers[name]["unit"]}
             for name, unit in original.items() if name in headers and headers[name]["unit"] != unit}
    return {"scope": "original_arterial_request_header_check", "status": "refused" if missing or units else "matched",
            "missing_required_tracks": missing, "unit_spelling_mismatches": units,
            "original_declaration_unchanged": True, "selected_record_values_decoded": 0,
            "arterial_pressure_import_admitted": False, "binding": json.loads(recording.binding_json)}


def retrospective_qc(path: str | Path) -> dict:
    """Whole-case inspection only. Never merge this output into replay inputs."""
    recording = _read_local(path)
    summaries = {}
    for name in TRACKS:
        records = [o for o in recording.observations if o.track == name]
        times = [o.source_timestamp_seconds for o in records]
        sorted_times = sorted(times)
        differences = [b - a for a, b in zip(sorted_times, sorted_times[1:])]
        positive = [d for d in differences if d > 0]
        summaries[name] = {"records": len(records), "first_timestamp_seconds": min(times),
            "last_timestamp_seconds": max(times), "duplicate_timestamps": len(times) - len(set(times)),
            "out_of_order_in_source": sum(b < a for a, b in zip(times, times[1:])),
            "nonfinite_values": sum(o.value_kind != "finite" for o in records),
            "min_positive_gap_seconds": min(positive) if positive else None,
            "median_positive_gap_seconds": statistics.median(positive) if positive else None,
            "max_gap_seconds": max(differences) if differences else None,
            "gaps_over_10_seconds": sum(d > 10 for d in differences),
            "gap_threshold_meaning": "descriptive interval count only; not a clinical missingness threshold"}
    return {"scope": "whole_case_retrospective_qc_not_replay_input", "binding": json.loads(recording.binding_json),
            "header": json.loads(recording.header_json), "track_headers": json.loads(recording.track_headers_json),
            "parse": json.loads(recording.parse_stats_json), "tracks": summaries, "bounds": LIMITS,
            "selected_records": len(recording.observations), "limitations": list(LIMITATIONS),
            "learning_contributions": {"component_fitting_examples": 0, "imitation_examples": 0,
                                       "RL_transitions": 0, "optimizer_updates": 0}}


class ObservedReplay:
    """Deterministic forward-only record release on the original recording clock.

    File packet ordinal breaks timestamp ties without discarding duplicates.
    Sorting only defines retrospective presentation order; source ordinal and
    timestamp bits remain attached. Source timestamp != proven device availability.
    """
    def __init__(self, path: str | Path):
        recording = _read_local(path)
        self.__events = tuple(sorted(recording.observations,
                                    key=lambda o: (o.source_timestamp_seconds, o.packet_ordinal)))
        self.__binding = recording.binding_json
        self.__clock = 0.0
        self.__cursor = 0
        self.__last: dict[str, NumericObservation] = {}
        self.__prefix = hashlib.sha256()
        self.__release(0.0)

    def __release(self, clock: float) -> list[dict]:
        released = []
        while self.__cursor < len(self.__events):
            event = self.__events[self.__cursor]
            if event.source_timestamp_seconds > clock:
                break
            record = event.to_dict()
            self.__prefix.update(_json(record) + b"\n")
            self.__last[event.track] = event
            released.append(record)
            self.__cursor += 1
        return released

    def advance(self, clock_seconds: float) -> dict:
        if isinstance(clock_seconds, bool) or not isinstance(clock_seconds, (int, float)) or not math.isfinite(clock_seconds):
            raise VitalDBError("Replay clock must be a finite number")
        if clock_seconds < self.__clock:
            raise VitalDBError("Replay clock cannot move backwards; reopen a snapshot")
        released = self.__release(float(clock_seconds))
        self.__clock = float(clock_seconds)
        frame = self.frame()
        frame["newly_released_records"] = released
        return frame

    def frame(self) -> dict:
        last = {}
        for name in TRACKS:
            event = self.__last.get(name)
            last[name] = None if event is None else {"observation": event.to_dict(),
                "source_record_age_seconds": self.__clock - event.source_timestamp_seconds,
                "source_record_at_this_replay_time": event.source_timestamp_seconds == self.__clock,
                "actual_measurement_age_unknown": True,
                "semantics": "last_recorded_observation; no continuous or current measurement inferred"}
        return {"schema": "resectionlab.vitaldb-observed-frame.v1", "binding": json.loads(self.__binding),
                "clock_seconds": self.__clock, "clock_origin_seconds": None,
                "time_basis": "unmapped_native_source_seconds",
                "clock_semantics": "unaltered native timestamp seconds; recording-start origin and observation latency unresolved",
                "missing_required_arterial_tracks": list(MISSING_ARTERIAL_TRACKS),
                "pressure_modality": "intermittent_noninvasive_cuff_pressure",
                "last_recorded_observations": last, "limitations": list(LIMITATIONS)}

    def visible_history(self) -> list[dict]:
        """Every released record, including ties and records at source time zero."""
        return [event.to_dict() for event in self.__events[:self.__cursor]]

    def snapshot(self) -> dict:
        return {"schema": "resectionlab.vitaldb-observed-snapshot.v1", "binding": json.loads(self.__binding),
                "clock_seconds": self.__clock, "released_records": self.__cursor,
                "released_prefix_sha256": self.__prefix.hexdigest()}

    @classmethod
    def reopen(cls, path: str | Path, snapshot: dict) -> "ObservedReplay":
        if not isinstance(snapshot, dict) or set(snapshot) != {"schema", "binding", "clock_seconds", "released_records", "released_prefix_sha256"}:
            raise VitalDBError("Invalid snapshot fields")
        if type(snapshot["released_records"]) is not int or snapshot["released_records"] < 0:
            raise VitalDBError("Invalid snapshot record count")
        replay = cls(path)
        if snapshot["schema"] != "resectionlab.vitaldb-observed-snapshot.v1" or snapshot["binding"] != json.loads(replay.__binding):
            raise VitalDBError("Snapshot source/role/declaration/code binding mismatch")
        replay.advance(snapshot["clock_seconds"])
        if replay.snapshot() != snapshot:
            raise VitalDBError("Snapshot cursor/prefix mismatch")
        return replay
