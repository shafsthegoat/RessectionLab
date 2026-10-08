# VitalDB recorded component replay candidate

This is a non-neurosurgical component-import/replay study, not glioma surgical
experience, anesthesia control, or an RL cohort. No scientific signal payload
has been acquired. Root prospectively selects case3 / subject2861 for DEVELOPMENT
import/replay; no existing patient assignment changes. All other VitalDB people
remain unopened and unassigned pending a prospective family-wide declaration.

The immutable [PhysioNet1.0.0 release](https://physionet.org/content/vitaldb/1.0.0/)
is open-access CC BY4.0. The case/subject projection gives6388 cases/6090 people;
subject2861 has only case3. The lowest case number containing the required live
inventory tracks was selected without inspecting clinical outcomes or signals.
The live API inventory is not proof of channel presence in the immutable file.

[0003.vital](https://physionet.org/files/vitaldb/1.0.0/vital_files/0003.vital):
6,537,712 bytes, published SHA256
`573db0941d580167833f84497f5d1c4a1391443eeec3f95852287ea09db024e4`.
HEAD returned200 without redirect; Last-Modified2022-08-17T13:16:32Z.
Its ETag is not a cryptographic hash. The [checksum manifest](https://physionet.org/files/vitaldb/1.0.0/SHA256SUMS.txt)
is562,636 bytes, SHA256
`874732d0548b5ba4de2efe851375c4f25f45488a2a19c771e2f9ab0288a5de1d`.
Clinical metadata SHA256 is
`7d6edb471e5eee3fde75e417084240c97bdbf6eff41cbd61e5dace44f1585ecf`;
only leading `caseid,subjectid` fields were inspected for identity grouping.
The track dictionary SHA256 is
`31c431642f91153461724375c05309d251ea63f206f3aeaed782ea4b9817400c`.

The initial numeric channels are `Orchestra/PPF20_RATE`, `PPF20_VOL`,
`RFTN20_RATE`, `RFTN20_VOL` (mL/hr, mL), and `Solar8000/ART_MBP`,
`ART_SBP`, `ART_DBP` (mmHg). Preserve native timestamps relative to recording
start, asynchronous series, duplicates and gaps. Pump-rate observations do not
prove exact circulation delivery or bolus timing; five-minute EMR anesthesia
boundaries do not establish device synchronization. Device CP/CE predictions
are not measured blood concentrations. The optional SNUADC/ART waveform is
not needed for the initial numeric importer.

## Parser inspection and implementation consequence

The [vitaldb1.7.2 MIT source archive](https://files.pythonhosted.org/packages/f4/1c/f48900276672017f7f72f3d2c0b94ac435c80fc0907ec663cedb6be76cf3/vitaldb-1.7.2.tar.gz)
is67,723 bytes, SHA256
`70e0ce784b13d52bbf6a315b742ab06d5ed0599b8b01f76182971a391dd64130`.
Independent read-only inspection authenticated the archive and read its MIT
license (copyright2021 VitalDB). No installation or execution occurred.
`vitaldb/utils.py` SHA256 is
`b0e88b1365c8d9a88814123c9a5b5a5d3c696a5c4849c7db3ebda476cc7134e3`.

Use only a locally authenticated PhysioNet file: `load_case(3, ...)` or
`VitalFile(3, ...)` fetches API dataset1.0.1, a different release. Native numeric
records are dictionaries with `dt` (source double seconds) and `val`; track
objects retain units, format, sample rate, gain and offset. Array/dataframe/sample
helpers quantize timestamps and can overwrite same-bin records. `dtstart` is
updated during selection and must not replace the source time origin.

The parser accepts suffix/wildcard selectors, ignores some malformed packets,
swallows EOFError and lets the constructor ignore a false parse result. Raw
packet type10 bypasses track filtering. `header_only` still traverses packets.
Therefore use a small audited local-file reader derived from the MIT source,
retaining its notice: exact track names/types/units; bounded packet lengths;
complete gzip validation; skip unrelated payloads without decoding; preserve
record ordinals and raw timestamps; report nonfinite/out-of-order records.
Do not infer doses, interpolate gaps, generate missing monitors or call a
successful constructor a complete import.

Next executable step: retain the prospective role/source declaration and pinned
parser source, then acquire the single6.54 MB file and validate its native
headers/records. Implement the common-clock replay only after those checks.
Recorded action/endpoint support and confounding remain separate RL gates.

Research failures retained: one release-page timeout, one live-inventory gzip
parse failure and two guessed GitHub source paths returning404. Subsequent
successful primary metadata/source reads resolved candidate size/parser access;
no agreement, outreach, outcome selection or scientific signal read occurred.
