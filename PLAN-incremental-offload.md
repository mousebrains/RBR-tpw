# Plan: incremental offload and one growing NetCDF per deployment

**Implemented 2026-09-28 on branch `incremental-offload` (P1-P4, P6; P5 bench validation pending). Departures from
the text below are listed in NOTES.md, "Incremental offload implemented".**

Draft 2026-09-27, revised through four self-review passes (recorded at the end; each finding names the
change it caused, and the text above the reviews is the revised version). The fourth pass, 2026-09-28, is
for Pat's point that the logger may be plugged into Ruskin and stopped, synced, erased or enabled between
offloads. Not implemented.

## Goal

Pat, 2026-09-27: instead of a new NetCDF per offload, one NetCDF per deployment that grows forwards. The
per-offload state that is now in global attributes (battery voltage, energy counter, memory used and
remaining, clock skew, NTP state, remaining-time estimates) becomes a time series in that file. Each
offload reads only the bytes the logger has added since the last one, from a stored offset.

## Evidence this rests on

- Every later download of the same deployment is a byte-for-byte extension of the earlier one: 12 of 12
  consecutive pairs of SN100689 images (fwtype 9, 2026-09-25/26). The 512-byte header is identical within a
  deployment in all 12 pairs and differs across every erase (6 of 6), at bytes 12-14 (enable time) and
  504-511. One logger, one firmware type: duets, concertos, Gen3 and Gen4 are unverified.
- `read data 1 <size> <offset>` (L2), `readdata` (Gen3) and `download` (Gen4) all take a byte offset, and
  `Link.read_data` already does. The resume code in `solo.download` already compares the first 512 bytes.
- Decode plus NetCDF write is seconds; the download is minutes (184 kB/s measured, 132 MB memory).
  Regenerating the NetCDF from the whole image costs nothing that matters.
- Ruskin's stop leaves the header alone and appends one event: in 37 of 37 stopped RBRsolo `.rsk` files (of
  111 solo files under `~/tpw/AEM1-G/data`) the header status word is still 0xFFFFFFFF and the last event is
  0x02 "stop command received". One file says `logging` yet ends with a stop event, unexplained. No file has
  a second 0x01 time-sync marker after the one written at enable (111 files, and the 18 bench images).
- The clock cannot be set while logging (Gen3: `E0105 command prohibited while logging`, L3 reference;
  Ruskin's own solo sequence sends `stop` before `now =`), and `enable` refuses unless memory is empty
  (`E0402`, solo and Gen3, NOTES). So a sync implies a stop, and a re-enable implies an erase.

## Decisions

D1. **Grow the raw image, regenerate the NetCDF.** The deployment's `.bin` is extended by each offload's
    new bytes. The NetCDF is rewritten from the whole image every time, with the existing temp-then-rename
    writer. No in-place NetCDF append, no unlimited dimension: HDF5 is not crash-safe, and sample times are
    not append-only (D5).
D2. **A deployment is identified by serial number and a tail check; the header is metadata.** The latest
    deployment on disk for this serial number is the only candidate. The last block of the image we hold
    (up to 68000 bytes ending at the stored offset) is read back from the logger and compared byte for
    byte. That costs one block read (about 0.4 s) and asks the only question that matters: does the logger
    still hold the bytes we have, at the same place. A mismatch, or a short read because the memory is now
    smaller than our image, means a new deployment; the old files are never touched. The 512-byte header is
    compared as well, but a header that changed while the data tail matched is the same deployment with
    changed logger state: the image's header bytes are refreshed from the logger, the change is recorded
    and warned about, and the offload continues. Mismatch fails safe: at worst a full download and a new
    file.
D3. **Names stay as they are.** The deployment is named after its first offload, `<SN>_<T0>`, exactly the
    stem an offload gets today. Later offloads of the same deployment rewrite `<SN>_<T0>.nc` and extend
    `raw/<SN>_<T0>.bin`; each offload still writes its own `raw/<SN>_<Tk>.json` record and `.log`
    transcript. A deployment offloaded once looks exactly like today's output.
D4. **The in-progress segment keeps the `.partial` mechanism.** The deployment image is only ever extended
    by a complete, CRC-checked segment, appended and fsynced before the record is written. Ctrl-C and
    reconnect resume the segment as they do now.
D5. **One clock skew per clock segment, chosen from all offloads of the deployment.** Today the skew
    measured at the offload re-times the samples on a reset clock in the last segment only. With several
    offloads, a reset segment is re-timed by the latest offload whose current clock segment it was, so a
    segment that ended before the last offload is no longer dropped. A single offload gives today's result.
D6. **Global attributes stay and the series is added.** The attributes keep meaning "state at the latest
    offload", so `rbr-rsk2nc` output and anything reading the attributes is unchanged. The new `offload`
    dimension carries the same quantities per offload.
D7. **Incremental reads for L2 first.** Gen3 and Gen4 get the growing file from the start but keep a full
    download each time, and the driver checks that the new image extends the old one. That check is the
    evidence needed to enable incremental reads there, and it costs nothing (the download happens anyway).
D8. **Not in this plan:** a per-logger file across deployments, a drift model that interpolates the skew
    along the deployment, adopting old-format records as a deployment's start. Reasons in "Deferred".

## Files and the record

```
OUTDIR/<SN>_<T0>.nc              regenerated at every offload of the deployment
OUTDIR/raw/<SN>_<T0>.bin         the memory image, extended by each offload (L2 dataset 1)
OUTDIR/raw/<SN>_<Tk>.json        one record per offload, k = 0, 1, 2, ...
OUTDIR/raw/<SN>_<Tk>.log         serial transcript of offload k
OUTDIR/raw/<SN>_<Tk>_configure.json   as today
OUTDIR/raw/.partial/<SN>.<T0>.<offset>.part   the segment being downloaded (deleted once recorded)
```

The record gains one key, and `raw` describes the image prefix this offload saw:

```json
"deployment": {
  "stem": "100689_20260925T194011Z",
  "header_bytes": 512,
  "header_sha256": "...",
  "tail_check": {"offset": 568908, "bytes": 68000, "ok": true},
  "header_changed": [],
  "offload_index": 3,
  "segment": {"offset": 636908, "bytes": 272, "sha256": "..."},
  "image_bytes": 637180,
  "image_sha256": "...",
  "download": "incremental"
},
"raw": {"file": "raw/100689_20260925T194011Z.bin", "bytes": 637180, "sha256": "..."}
```

`download` is `incremental`, `full` (nothing to extend, or `--full-download` on a new deployment) or
`full-verified` (`--full-download` with an existing image whose prefix matched). Gen3 records list the
datasets as today, each with `offset`, `bytes` and `sha256` of the segment; datasets 0 (events) and 2
(header) are re-read whole, rewritten atomically, and have offset 0. Records without a `deployment` key are
old-format and still rebuild as before.

## One offload, step by step (L2)

1. Identify, NTP, clock skew, snapshot: unchanged. `total = meminfo used` at the snapshot.
2. Read the header: the first `min(512, total)` bytes (the read `solo.download` already makes).
3. Find the deployment: load `raw/<SN>_*.json`, keep the records with a `deployment` key, group by `stem`,
   take the stem with the highest `offload_index` (ties: latest `offload_started`). Its latest record
   gives `R = image_bytes`, `H = image_sha256` and the previous header. No record: new deployment,
   `stem = <SN>_<now>`, `R = 0`, go to step 7.
4. Check the image on disk, `L = size of raw/<stem>.bin`:
   - `L == R` and `sha256(file) == H`: normal.
   - `L > R` and `sha256(first R bytes) == H`: a crash between the append and the record. Truncate to `R`.
     The segment's `.part` file, if it survived, resumes; otherwise the segment is read again.
   - anything else (`L < R`, hash mismatch, file missing): the image is not trustworthy. Log an error
     naming the file, start a new deployment stem with a full download, leave the old files alone.
5. Tail check (D2): `n = min(68000, R)`; read logger bytes `[R - n, R)` and compare the data part,
   `[max(R - n, 512), R)`, with the file. A short read (`total < R`) or a difference: a different
   deployment (erased and enabled in Ruskin or by `--configure`, or a memory we cannot explain). Log it
   with the first differing offset, new stem, full download. Equal: the offset is confirmed against the
   data itself. Then compare the header from step 2 with the image's first 512 bytes: differing bytes are
   refreshed in the image, listed in the record as `header_changed`, and warned about. With `R <= 512`
   there is no data part, and a differing header is a new deployment.
6. `total == R`: nothing new. Skip the download; the record, the series point and the NetCDF are still
   written (step 9 onwards).
7. Download bytes `R..total` into `.partial/<SN>.<T0>.<R>.part`, resuming if it exists (a `.part` with any
   other stem or offset is deleted). The part file holds segment bytes only; `solo.download` skips its
   head comparison when `start > 0`, since steps 3-5 established the deployment.
8. Append the segment to `raw/<stem>.bin`, flush, fsync. Compute `image_sha256` of the whole file.
9. `after` snapshot, remaining-time estimate and alarms: unchanged.
10. Write `raw/<SN>_<Tk>.json` atomically, with `deployment` and `raw` as above. Only now delete the
    `.part`. Console: `incremental: 272 new bytes from offset 636908 (deployment 100689_20260925T194011Z,
    offload 3)`, or `new deployment` with the reason when one was started because of a check failure.
11. `--configure`: unchanged. It erases the memory, so the next offload starts a new deployment. The note
    that bytes logged during the download are not saved stays true only in this case; without
    `--configure` they arrive next time, and the message says so.
12. Regenerate `<stem>.nc` from `raw/<stem>.bin` and every record of the stem in `offload_index` order
    (the NetCDF write already runs on the main thread; unchanged).

`--full-download` (new flag): read `0..total` regardless. If a deployment matched, compare the first `R`
bytes with the image on disk; equal means `download = full-verified` and only bytes `R..total` are
appended; unequal is an error that names the first differing offset, leaves the deployment alone and starts
a new stem. This is both the escape hatch and the bench validation tool.

## Ruskin between offloads

Pat, 2026-09-28: between two offloads the logger may be plugged into Ruskin and stopped, synced, erased
and/or enabled. What each action does to the memory and the clock, and what catches it:

| Ruskin action | Memory and clock | Caught by | Outcome |
|---|---|---|---|
| Connect, download | nothing changes (`lock OFF`, reads, `lock on`) | tail check passes | incremental as usual |
| Stop | one 0x02 event appended; header unchanged (37 of 37 files); status `stopped` | tail check passes; the new bytes are the event | incremental; no remaining-time estimate; the stop is in `events` and noted in `history` |
| Sync (set clock) | only possible after a stop; no samples follow until an erase and enable; the next skew is measured on the new clock | skew jump between offloads, status `stopped` | reset-clock samples keep the pre-sync offload's skew; samples logged between that offload and the stop are dropped with a note (time resolution, below) |
| Erase | memory empty, header gone | `total == 0` | nothing to download; the deployment's files stay as they are |
| Erase and enable | new header (enable time), new data | tail check fails (short read or different bytes) | new deployment stem, full download |
| Settings changed while stopped (start, end, period) | take effect only at enable, which needs an erase | settings diff | warning naming the setting; the deployment continues |
| Firmware update | memory erased; version and possibly fwtype change | tail check fails; `id` differs | new deployment; the driver is chosen per offload as today |

Sanity checks on every offload, each producing a warning in the record, on the console and in the NetCDF
`history` and `warnings`:

- Settings diff against the deployment's previous record: sampling mode and period, start and end time,
  channel count, types and coefficients, firmware version, memory format.
- Status transition (`logging` to `stopped`, `finished` or `pending`), with the event that explains it when
  one was appended (0x02 stop, 0x0C end time reached).
- Skew jump: `|skew_k - skew_(k-1)|` beyond `2 s + 50 ppm x elapsed` means the clock was set or reset
  between the offloads. Explained when the new bytes hold a reset event; otherwise a sync. Recorded as
  `clock_set_detected` on the `offload` series.
- Memory smaller than our image, or a changed header: steps 4 and 5.
- New event types in the new bytes, listed by name.

What makes this tractable is the logger's own rule: the clock cannot be set while logging, and enable needs
an empty memory. Whether Ruskin 2.26.1 offers a clock set outside its enable sequence, and whether the solo
writes a 0x01 marker on `now =` while stopped, are P0 bench questions; the handling above does not depend
on the answers.

## Time resolution across offloads (rawbin)

Today `resolve_time_arrays` re-times the reset-clock samples of the last clock segment by the single
offload's skew, checked against that offload's time. Generalize:

- `Decoded` gains `set_end_byte`, the byte offset just past each sample set's last word (events and dropped
  bad-event words included). `Event` already has `offset`.
- Each offload k becomes an `OffloadView(samples_seen, events_seen, unix_ms, skew_s)`: for L2,
  `samples_seen = count(set_end_byte <= image_bytes_k)` and `events_seen = count(event.offset + 8 <=
  image_bytes_k)`; for Gen3 EasyParse, `dataset1_bytes_k // record_size` and `dataset0_bytes_k // 16`; Gen4
  the same from its data and events objects. Records without a measured skew give no view.
- `clock_segments` also returns each segment's start marker: the reset event that began it, or, for a
  segment found by a backward step of the sample times, its first sample index. Offload k's current
  segment is the last segment whose start marker it had seen (event index < `events_seen`, or sample index
  < `samples_seen`); segment 0 is always seen.
- A reset-clock segment's candidates are the offloads whose current segment it is. The latest candidate's
  skew and time are used. The sanity checks become: the last sample that offload saw, if it saw any of the
  segment, lands no later than the offload plus 5 s; the segment's first sample lands after the previous
  kept sample; its last sample lands before the next segment's first sample after that segment's own
  correction; nothing lands before 2001. A segment failing a check is dropped with a note, as an
  uncorrectable one is today. The final strict monotonic check stays as the last guard.
- Clock runs also end at a clock set. Two signals: a 0x01 time-sync marker that is not the deployment's
  first event (the decoder already re-anchors there), and a skew jump between consecutive offloads (Ruskin
  section). A marker locates the boundary exactly. A skew jump between offloads k-1 and k does not, so the
  boundary is placed at `samples_seen_(k-1)`: samples offload k-1 saw keep its candidates, and later
  samples of the same reset-clock run get no candidate from before k. With the stop-before-sync rule those
  are only the samples logged between offload k-1 and the stop; they are dropped with a note that says so.
  Offload k's own skew is on the new clock and never applies before the boundary.
- A candidate whose correction fails the sanity checks is skipped in favour of the previous candidate; the
  segment is dropped only when no candidate passes.
- One offload reproduces today's result exactly: the last segment's only candidate is that offload, and
  earlier segments have none. A regression test asserts this on the existing fixtures.
- Why the latest candidate: the skew is exact at its own measurement time and the error grows with the
  clock's drift over the interval back to each sample. This is today's behaviour with one offload;
  interpolating between offloads is the deferred drift model, and the series this plan writes is what it
  needs.

Timestamped families (Gen3 EasyParse, Gen4) already go through `resolve_time_arrays` and use the same
views.

## The offload series in the NetCDF

Dimension `offload_time` (length = number of records) with the coordinate variable of the same name (int64
ms since 1970, `standard_name: time`, the same `units_metadata` as `time`, no `axis`; `time` keeps
`axis: T`), so xarray and CF tools index it without a `coordinates` attribute. Variables, all on
`offload_time`. A prototype built from SN100689's five real offloads is in
`output/prototype/100689_20260925T211228Z.nc` (2026-09-28); it passes the cf:1.11 checker.

| Variable | From the record | Notes |
|---|---|---|
| `offload_time` | `offload_started` | coordinate |
| `offload_image_bytes` | `deployment.image_bytes` | bytes of memory held after this offload |
| `offload_segment_bytes` | `deployment.segment.bytes` | bytes read at this offload |
| `offload_samples` | computed | sample sets seen by this offload (`samples_seen`) |
| `battery_voltage` | `after.power.battery_voltage_V` | read on USB power: unloaded cell |
| `battery_energy_remaining` | `after.power.energy_remaining_J` | the logger's counter, NaN when absent |
| `memory_used`, `memory_remaining` | `after.meminfo` | bytes |
| `clock_skew`, `clock_skew_uncertainty` | `clock_skew` + `host_ntp` | logger minus UTC, s; NaN without NTP |
| `clock_skew_vs_host` | `clock_skew.skew_vs_host_s` | |
| `host_ntp_offset`, `host_ntp_uncertainty` | `host_ntp` | |
| `sampling_days_remaining`, `energy_days_remaining_modelled` | `remaining_time` | NaN when not estimated |
| `logger_status`, `power_source`, `offload_port`, `offload_tool_version` | record | strings |
| `clock_set_detected` | computed | 1 when the skew jumped since the previous offload with no reset event to explain it |
| `header_changed` | `deployment.header_changed` | number of header bytes that changed since the previous offload |

The logger writes its own energy series on Gen3: event 0x27 "energy used marker" carries `powerinternal used`
as a float32 in joules, at enable and then every 33.8 h (NOTES, 2026-09-28; 2 concerto³ loggers, 24 markers).
Today the NetCDF drops event payloads. Add `event_payload` (uint32, on `event`) for the EasyParse and Gen4
event tables, and `energy_used_marker` (float64 J, NaN except on 0x27 events) so the logger's accounting
sits beside the per-offload counter without a second dimension. L2 events have no payload; nothing changes
there.

`history` gets one line per offload, oldest first: time, port, tool version, bytes added, download kind.
Global attributes are computed from the latest record as today (D6). `time_coverage_*` span the whole file.
`rbr-rsk2nc` writes the same layout from its single pseudo-record, so the series has length 1 there.

## Rebuild

`--rebuild RECORD.json`: a record with a `deployment` key rebuilds `<stem>.nc` from the deployment's raw
files and all records of that stem on disk, in `offload_index` order. The latest record's hashes are
verified against the files (for Gen3 the event and header datasets are rewritten each offload, so earlier
records' hashes no longer describe the files); earlier records contribute their skew, their health values
and their `image_bytes`, which must be non-decreasing along the sequence. Several records of one stem on
the command line rebuild it once. Old-format records rebuild `<record stem>.nc` from their own `.bin` as
now.

## Progress meter on a terminal (Pat, bench 2026-09-26; added to this plan 2026-09-28)

Today the download logs a line every 10% (`_progress_logger` in `cli.py`, called once per 68000-byte block,
about three times a second at 184 kB/s), and with several loggers a summary line every 30 s. On a terminal
this becomes one status line at the bottom, rewritten in place:

```
[SN100689@usbmodem101] downloading 41.3 of 132.1 MB [=======>              ]  31%  184 kB/s  8:12 left
```

- **Owner: `Console`.** It already owns the terminal, holds output while a question is up, and detects a
  TTY (the `color` test, with ANSI enabled on Windows). `Console.status(text | None)` keeps one transient
  line: it is drawn with `\r` and clear-to-end-of-line, every ordinary `write()` clears it first and redraws
  it after, `ask()` clears it before the prompt and redraws after the answer, and `status(None)` clears it
  for good. `status_enabled` is the TTY test and a constructor argument, so tests and non-TTY runs (cron,
  a redirected log) never see `\r`. The text is cut to the terminal width minus one column.
- **Source: `_stage()`.** Every stage already reports itself (`identifying`, `measuring clock skew`,
  `downloading 31%`, `writing NetCDF`). `_stage()` also feeds `Console.status()`: with one logger in
  `_stages` the line is that logger's stage; the download stage carries the bar, bytes, rate and ETA that
  `_progress_logger` already has per block. With more than one logger in `_stages`, the line is
  `_stage_summary()` (`SN100689 downloading 31%; SN100690 measuring clock skew`), refreshed on every stage
  change, so a second logger appearing mid-download turns the bar into the summary instead of two bars
  fighting over one line. The 30 s summary log line is then not written on a terminal.
- **The 10% lines stay in the session log.** On a terminal they would duplicate the bar, so
  `_progress_logger` logs them at DEBUG when the status line is enabled and at INFO otherwise. The
  session log gets them either way.
- **Rate and ETA** are what `solo.download` reports: the mean since the segment started. No smoothing.
  With incremental downloads the total is the segment, not the memory, so a 272-byte catch-up shows one
  block at 100%.
- **Ctrl-C and exit:** `run()` clears the status line in its `finally`, so the "Stopped." and "done with
  ..." lines are never printed over it.

## Changes by module

- `rawbin.py`: `Decoded.set_end_byte`; `OffloadView`; `clock_segments` returns start markers and also ends
  a run at a later 0x01 marker; `resolve_time_arrays(..., views)` with the skew-jump boundary and the
  candidate fallback. `equations.decode_l2` (sectioned header, duet and concerto) computes
  `set_end_byte` the same way.
- `ncwrite.py`: `write_netcdf(image, records, path)` and `write_values_netcdf(..., records, ...)` take the
  ordered list; metadata from `records[-1]`; `_offload_series(nc, records, views)`; `history` from all
  records; `_event_variables` gains the payload and the energy marker for timestamped families (the
  `events` tuples already carry the payload as far as `write_engineering`, which drops it).
  `skew_vs_utc(record)` is reused per record.
- `solo.py`: `download(link, total, part_path, progress, dataset, l3, start=0)`: reads `start..total`; the
  part file holds segment bytes; no head comparison when `start > 0`.
- `drivers.py`: `Driver.header(link, total) -> bytes` (L2: first 512 bytes of dataset 1; Gen3: dataset 2;
  Gen4: the first dataset's `meta`); `Driver.tail_check(link, image_path, R) -> bool`;
  `Driver.download(..., resume: DeploymentState | None)` returning segments `{name: (offset, bytes)}`; L2
  incremental, Gen3/Gen4 full plus the prefix check (D7). Gen4's own per-object partial files stay as they
  are.
- `cli.py`: `_find_deployment(rawdir, sn, header_sha) -> DeploymentState | None` (steps 3-5), the append
  (step 8), the record keys (step 10), `--full-download`, `_rebuild` grouping by stem, the console lines,
  the "not saved" message, and `_compare_with_previous(record, previous)` for the between-offload checks
  (Ruskin section).
- `rsk.py`: passes `[record]`.
- `console.py`: `Console.status()`, `status_enabled`, the clear-and-redraw in `write()` and `_answer()`,
  width from `shutil.get_terminal_size()`.
- `cli.py` (progress meter): `_stage()` calls `Console.status()`; `_progress_logger` builds the bar text and
  picks the log level; `run()` clears the line on exit and skips the 30 s summary line on a terminal.
- `tests/fakelogger.py`: `FakePort.grow(more: bytes)` rebinds both `image` and `datasets[1]` (bytes are
  immutable, so `+=` on one does not reach the other); `erase()` for L2 fakes that are not writable;
  `FakeSoloWritable.stop` appends a 0x02 event to the image, as the real logger does, so Ruskin's actions
  are driven by sending the fake the same commands Ruskin sends.
- `README.md`: Offload section (growing file, `--full-download`, what a record holds, rebuild); the file
  table; the migration note. `NOTES.md`: a dated entry with the evidence table and the decisions.

Migration: an OUTDIR with old-format records starts a new deployment stem with one full download per logger.
A leftover old-style `.partial/<SN>.bin.part` from an interrupted download is not resumed; it is deleted
and that download is repeated.

## Tests

1. Two offloads of one fake solo with growth in between: the second sends `read data` only for the header,
   the tail check and offsets from `R`; one `.nc`, `len(time)` = all samples, `offload` dimension 2, two
   `history` lines, records 0 and 1 name the same stem, `raw/<stem>.bin` equals the fake's image.
2. Nothing new (no growth): header and tail-check reads only; record and `.nc` rewritten; series length 2.
3. Erase between offloads (new header): new stem; the old `.nc` and `.bin` are byte-identical before and
   after.
4. Crash windows: `.bin` longer than the record (truncated, then extended correctly); `.bin` shorter or hash
   wrong (error logged, new stem, old files untouched); `.part` with a stale offset (deleted); tail check
   failing on a fake whose image was replaced under an unchanged header (new stem, error names the offset).
5. Ctrl-C mid-segment then reconnect: the existing resume test, with the segment part name and `R > 0`.
6. Reset clock spanning two offloads: one skew for the segment, no step at the boundary, times strictly
   increasing. A reset segment that ended before offload 2: kept with offload 1's skew (today it would be
   dropped). An offload that saw the reset event but no sample of the new segment (long period): counted
   as the new segment's candidate, not the old one's. The regression test that one record reproduces
   today's `resolve_time_arrays` output.
7. `--full-download` on a matched deployment: `full-verified`, only the tail appended; on a mismatching
   prefix: error, new stem.
8. `--rebuild` with a deployment record, with several records of one stem, and with an old-format record.
9. `rbr-rsk2nc` output unchanged apart from the length-1 series and the event payloads; the CF compliance
   test (`conftest.py`, cf:1.11) passes on a two-offload file. The concerto³ fixture with 0x27 events gives
   `energy_used_marker` values equal to the float32 payloads.
10. Gen3 fake, two offloads: full download both times, prefix check passes, growing file; a fake that
    rewrites dataset 2 mid-deployment is reported as a new deployment.
11. Duet fake (sectioned header): identity, tail check and `set_end_byte` through `equations.decode_l2`.
12. Ruskin stop between offloads (`stop` to the writable fake, which appends the 0x02 event): same stem,
    the event in the new bytes, status `stopped`, no remaining-time estimate, the transition noted.
13. Stop, then `now =` (sync) with a reset-clock run in progress: the second offload measures a jumped
    skew; the run keeps offload 1's skew; samples logged between offload 1 and the stop are dropped with
    the note; `clock_set_detected` is 1 on the second point.
14. Stop, sync, `memclear`, `enable`: new stem, old files byte-identical.
15. Header changed with the data tail intact (a fake that rewrites a header word): same stem, header
    refreshed in the image, `header_changed` in the record, warning in `history`.
16. Settings changed between offloads (end time): warning naming the setting; the deployment continues.
17. Progress meter, `Console` with a stream whose `isatty()` is true: the status line is drawn with `\r`, a
    `write()` while it is up clears it and redraws it after the line, `ask()` clears it before the prompt,
    `status(None)` clears it; the text is cut to the terminal width.
18. Progress meter, non-TTY stream (the existing `rig` fixture): no `\r` anywhere in the console text, the
    10% lines are on the console as today; on a TTY console they are in the session log at DEBUG only.
19. Two loggers on a TTY console: the status line is the stage summary, not a bar, and the 30 s summary
    line is not logged to the console.

Bench (acceptance, SN100689): offload, wait a few minutes, offload again, then `--full-download`; expect
`full-verified`, and `<stem>.nc` from that run identical to the incremental one apart from the series and
`history`. Repeat across a `--configure` erase. If a duet or concerto is on the bench, the same.

## Phases

- P0 Evidence: L2 growth and stop are done (above). Bench, SN100689 in Ruskin: connect only; stop; a clock
  set outside the enable sequence, if Ruskin offers one (does the solo refuse `now =` while logging, and
  does it write a 0x01 marker while stopped); erase; enable. For each, an offload before and after, with the
  header diff, appended events, `used` delta and skew jump recorded in NOTES. Gen3/Gen4 header stability
  comes from P4's prefix check on real loggers.
- P1 `rawbin` and `ncwrite` with a list of records; one-record output unchanged (tests 6, 9, 11).
- P2 `cli` deployment bookkeeping and L2 incremental reads (tests 1-5, 7).
- P3 Rebuild, `rbr-rsk2nc`, README, NOTES (test 8).
- P4 Gen3/Gen4 growing file with the prefix check (test 10).
- P5 Bench validation; enable Gen3 incremental reads only after P4 has shown constant headers and prefix
    growth on a real logger.
- P6 Progress meter (tests 17-19). Independent of P1-P5; it can go first, since it is the piece the bench
  sees most, or last.

Effort, a guess: P1 1 d, P2 1.5 d, P3 0.5 d, P4 0.5 d, P5 0.5 d bench plus waiting, P6 0.5 d.

## Deferred, and why

- Per-logger series across deployments: a derived product of the records; can be made any time by
  concatenating the `offload` series of each deployment file.
- Drift interpolation: needs two or more skews per deployment, which this plan starts collecting. It changes
  the science product, so it should be a flagged option with its own validation.
- Adopting an old-format full image as a deployment's start: saves one full download per logger, once.
  Not worth the extra matching logic.

## Risks and open questions

- Gen3 dataset 2 and Gen4 `meta` are assumed constant within a deployment; unverified live (D7 handles it).
- A clock set is only possible while stopped, and no samples follow it without an erase. The remaining
  exposure is a reset-clock run whose only offload came after a sync: those samples have no recoverable
  time and are dropped with a note, as today.
- The regenerated write of a full 132 MB solo image (33 M samples) runs on the main thread and holds other
  loggers' prompts for its duration, as a full download's write does today.
- Record format: records gain keys; old records still read. Bump the version to 0.2.0.

## Self-review 1: data integrity (what can corrupt or lose data)

- F1.1 A header-only identity is weaker than it looks. A logger whose clock restarted at 2000-01-01 before
  enabling gets a small enable time, and two such deployments with the same period, start and end could
  share a header. Header bytes 504-511 might also change mid-deployment on a model not yet seen, which would
  only cost a full download, but a collision the other way would append one deployment to another.
  Change: the tail check (D2, step 5). One block read confirms the logger holds the exact bytes at the end
  of our image, and a short read catches `total < R` without a separate invariant.
- F1.2 An old-style `.partial/<SN>.bin.part` holds an image from offset 0 and would be misread as a
  segment. Change: the offset is in the part file's name and a part with any other name is deleted
  (step 7); migration note.
- F1.3 Gen3's event and header datasets are rewritten each offload, so an earlier record's hashes stop
  matching the files and a rebuild of an earlier record would fail its checksum. Change: rebuild verifies
  the latest record only and takes skews, health values and `image_bytes` from the earlier ones (Rebuild).
- F1.4 A crash after the record is written but before the `.part` is deleted leaves a part with offset
  `R_old`; the next offload's `R` is larger. Confirmed handled by the offset in the name (step 7), no change.
- F1.5 Every failure path (bad image, tail mismatch, memory shrank, prefix mismatch on `--full-download`)
  ends the same way: error, new stem, old files untouched. Confirmed consistent; wording aligned in
  steps 4, 5 and the flag.

## Self-review 2: timing and the science product

- F2.1 The draft's candidate rule ("offload happened between the segment's first byte and the next reset
  event's byte") had no equivalent for Gen3, whose events live in another dataset, and a sample-index
  version has a hole: an offload right after a reset, before the first new sample (long periods), would be
  taken as a candidate for the old, dead clock. Change: per-offload views with `samples_seen` and
  `events_seen`, segment start markers, and "the offload's current segment" as the rule; test 6 gains that
  case.
- F2.2 The existing 5 s sanity check compares the segment's last sample with the offload time. With several
  offloads, a segment may continue after the offload that corrects it, so its last sample legitimately
  lands after that offload. Change: the check uses the last sample the offload saw.
- F2.3 Two adjacent corrected segments can overlap, which today would make the final monotonic check refuse
  the whole file, and for a growing file that would block a deployment's NetCDF for good. Change: a
  two-sided per-segment check that drops the offending segment with a note; the final check stays as the
  guard.
- F2.4 Which candidate when several offloads sit in one clock segment: the latest, for the reasons now
  written under "Why the latest candidate". A hypothesis, stated as one: drift of the order of a second a
  day is common for logger RTCs, so the choice matters at the seconds level over weeks, and the series is
  how to measure this logger's actual drift before deciding on interpolation.
- F2.5 Samples partially inside the image at an offload count as unseen (`set_end_byte <= image_bytes`).
  Confirmed correct, no change.

## Self-review 3: scope, compatibility, tests

- F3.1 `--full` reads like "full erase" beside `--configure`. Change: `--full-download`.
- F3.2 Test 2 claimed no reads beyond the header; the tail check is a read. Change: tests 1 and 2 name the
  tail-check read.
- F3.3 The draft placed `decode_l2` in `rawbin.py`; it is in `equations.py`. Change: module list, test 11.
- F3.4 `FakeSolo` keeps `image` and `datasets[1]` as two names for one bytes object; growing one by `+=`
  rebinds only that name. Change: `grow()` rebinds both, stated in the module list.
- F3.5 Nothing said what the operator sees. Change: the console lines in step 10.
- F3.6 The CF compliance test exists (`conftest.py`, IOOS checker at cf:1.11) and covers the new series
  once a two-offload file is a fixture. Confirmed, named in test 9.
- F3.7 Gen4 has its own download with per-object partial files and is untested on hardware. Confirmed the
  plan leaves it on the Gen3 pattern (full download, prefix check) and does not touch its partial files.
- F3.8 Does the plan deliver the three things asked for? One growing file per deployment: D1, D3, step 12.
  Attributes as a series: the `offload` dimension, D6. Only the new bytes, from a stored offset: steps 3-8,
  with the offset being the previous record's `image_bytes`. Confirmed.

## Self-review 4: Ruskin between offloads (Pat, 2026-09-28)

- F4.1 The draft made the header the deployment's identity and a header change a new deployment. A stop, a
  finished schedule or a status word could in principle rewrite header state without touching the data,
  and that would have split one deployment into two files. Change: D2 makes the tail check the identity
  and a header change a recorded, warned state change; step 5 compares the data part of the tail
  separately from the header. Evidence gathered for this pass: a stop leaves the header status word alone
  in 37 of 37 files, so the case is rarer than feared, and the change costs nothing.
- F4.2 A sync after a reset-clock run poisons the next offload's skew for that run. The draft's "latest
  candidate wins" would have applied it, failed the "not before 2001" check and dropped the run, losing
  samples an earlier offload could time. Change: the skew jump ends the clock run at `samples_seen_(k-1)`,
  and a failing candidate falls back to the previous one.
- F4.3 The candidate rule depends on a sync being impossible while logging. The L3 reference says so
  (`E0105`), Ruskin's solo sequence stops first, and `enable` needs an empty memory (`E0402`). Change:
  stated as the rule in the Ruskin section, with the two open questions on the P0 bench list rather than
  assumed.
- F4.4 Nothing compared an offload's settings and status with the previous one. Change: the settings diff
  and status-transition checks, cheap and general, warned rather than acted on.
- F4.5 A stopped logger gets no remaining-time estimate and today's alarm logic already skips it (`status`
  not in logging, pending, gated). Confirmed; test 12 pins it.
- F4.6 The one `.rsk` file marked `logging` that ends with a stop event is unexplained. It does not change
  the design (the event is in memory either way) and is kept in the evidence rather than dropped.

## Self-review 5: progress meter

- F5.1 The 10% lines and a bar on the same terminal say the same thing twice. Change: DEBUG on a terminal,
  INFO otherwise; the session log keeps them in both cases.
- F5.2 A second logger plugged in during a download would leave two workers rewriting one line. Change: the
  line shows the stage summary whenever more than one logger is in `_stages`; the per-logger bar is the
  single-logger case only, which is what was asked for.
- F5.3 A prompt typed over a status line is unreadable, and an alarm banner printed over it loses its first
  characters. Change: `write()` and `_answer()` clear the line first and redraw after; the banner path goes
  through `write()` already.
- F5.4 The NetCDF write runs on the main thread inside `serve()`, so no progress callbacks arrive during
  it and the line just says `writing NetCDF`. Confirmed acceptable: the write is seconds, up to a minute on
  a full memory, and the stage name is already set before it starts.
- F5.5 Windows: `\r` and clear-to-end-of-line need ANSI processing, which `_enable_windows_ansi` already
  turns on for `color`; `status_enabled` follows `color`. Unverified on a real Windows console, as the
  colour path is.
