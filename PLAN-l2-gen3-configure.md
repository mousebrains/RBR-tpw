# Plan: configure, and offload incrementally, the RBRduet, RBRconcerto and RBRconcerto³

Status (2026-11-26): phase A built on branch `l2-gen3-configure` and bench-tested on both duets and one concerto (below); phase B next.

`--configure` covered only the RBRsolo (fwtypes 9 and 0), and only RBRsolo memory was read incrementally (new
bytes only). Everything else was read in full at every offload. A duet holding 133 MB takes about 15 minutes. The
work goes in three steps, each bench-tested before the next:

| Phase | Model (fwtype) | Command set | Configure | Incremental offload |
|---|---|---|---|---|
| A1 | RBRduet (102) | L2 | this branch | this branch |
| A2 | RBRconcerto (103) | L2 + twist activation | this branch | this branch |
| B | RBRconcerto³ (104) | Gen3 (L3 reference) | to do | to do (per dataset `readdata` with offset) |

## Evidence the write sequences rest on

- **Ruskin 2.26.1's serial logs** (`~/Ruskin/logs/ruskin_serial.log*`, 2026-09-10..25). Split at Ruskin's wake
  (`\r A \r`) into connections, each labelled by its `id` reply, they show:
  - 5 duet enables (fw 3.220): `lock OFF`, `stop` (if logging), `permit settings` + `settings offsetfromutc =
    0.0`, `now = <UTC>`, `powerstatus remaining = <its own value>`, `verify` (-> `E0402 memory not empty, erase
    first, verify = stopped`), then `enable erasememory = true` (-> `E0401 estimated memory usage exceeds
    capacity, enable = logging`, 0.4-0.6 s), and `lock on`. Ruskin also rewrites unchanged `settings` values.
  - 10 concerto enables (fw 1.460): the same, plus `twistactivation state = on` when it was off and `sampling
    period = 250` / `= 1000` when the period changed. Ruskin sends `sleep` and, at once, `enable erasememory =
    true, memformat = RAWBIN00` (-> `enable = gated`: twist activation on, twisted off).
  - Gen3 enables (concerto³, fw 1.158): `clock datetime = <UTC>, offsetfromutc = 0.0`, `deployment endtime = ..,
    starttime = ..`, `permit command = settings`, `memformat newtype = CALBIN00`, `verify` (-> `E0402`), `enable
    erasememory = true, memformat = CALBIN00` (-> `enable status = gated, warning = none`); `disable` stops.
- **Ruskin 2.26.2's `PhysicalL2`** (bytecode, javap): `setSampling` writes one `sampling k = v, ...` command
  holding only the fields that change (mode, period, burstlength, burstinterval). `eraseMemory` sends `permit
  memclear` + `memclear` and checks `used = 0`. `setTwistActivation` writes `twistactivation state = ...`.
- **L3 command reference rev L, section 3.9** (L2 dialect): `permit memclear` / `permit = memclear`,
  `enable erasememory = true` with the `E0401 ..., enable = logging` reply, `stop` / `stop = stopped`.
- **The loggers' own replies** in the offload transcripts: `sampling ... userperiodlimit = N` (duets 500,
  concertos 167, concerto³s 31 ms), `twistactivation state = on, location = off`, `settings ... offsetfromutc =
  +0.00` on every duet and concerto seen.

## Phase A: what is built (RBRduet and RBRconcerto)

1. **Paced clock-skew probe** (`solo.measure_clock_skew(pace_s=...)`, 102/103 only). Duets and concertos drop a
   command now and then (no reply at all). Unpaced, the concertos dropped 19 of 1273 `now` polls (1.5 %). A
   paced probe (>= 8 ms after each reply, 0.1 s timeout, no re-send) lost 0 of 946 on the same two loggers. A
   drop cost a 3 s re-send, which left n = 0 skews, and one bracket that spanned it was +/-1.68 s wide. Now a
   poll waits 10 ms after the last reply, gets 0.1 s, and is not re-sent. A drop clears the bracket and
   discards any late reply.
2. **A gap before every command** to these two models (`Link.min_gap_s` = `Driver.command_gap_s` = 10 ms), on
   the hypothesis (~90 %, not isolated) that pacing is what the probe got right.
3. **Incremental offload of sectioned images.** The identity check is the RBRsolo's: the first and the last
   68 kB block of the held image are read back and compared, with the header at its own length (715, 939 or
   1037 bytes). Only the new bytes are then read. Sectioned images had been read in full and checked to
   extend the saved one since PR #13's review (finding 11). On hardware every one of those reads did extend
   it: 15 concerto and 2 duet offloads. The one mismatch followed a re-enable.
4. **`--configure` for 102/103** (`configure._apply_l2`), Ruskin's sequence, writing only what changes:
   - `stop` only if the logger is logging, pending or gated;
   - `settings offsetfromutc = 0.0` after the clock set, whenever the offset does not read zero (a clock write leaves it
     `unknown`: see the bench results), checked in the readback;
   - `starttime` / `endtime` / `sampling period = N` / `twistactivation state = ...` only if they differ;
   - erase and enable together as `enable erasememory = true` (concerto: `, memformat = RAWBIN00`); memory used
     must then be less than before;
   - erase without enable (`--no-enable`): `permit memclear` + `memclear`, then `used = 0`.

   It does not copy Ruskin's rewrites of unchanged settings or of the duet's energy counter, nor the `sleep`
   before a concerto's enable. The enable saves the settings anyway.
5. **Lost replies during configure.** A write with no reply is read back. If it took effect, it is accepted
   (only the reply was lost); if not, it is sent once more. The unlock key is re-sent once with a fresh
   challenge. The clock write is never re-sent: the skew is re-measured, and that attempt does not move the
   lead. A lost enable is judged by the status and memory read back.
6. **Periods.** On 102/103, whole seconds, or 1000/n ms (rounded half up) for a whole rate of n Hz, no shorter
   than the logger's `userperiodlimit`. `--rate-hz` rounds half up (16 Hz -> 63 ms, as the Gen3 list has it).
   The 1000/n rule is an assumption: it fits the Gen3 list (`500|250|125|63|42|31`) and Ruskin's 250 ms on a
   concerto. The readback after the write is the check.
7. **Ready to deploy** includes `gated`: enabled, waiting for its twist. `--twist-activation on|off` (concerto
   only); default: leave it as it is.
8. **Not supported on 102/103:** `--fresh-battery` / `--used-battery` (the counter's units and a new cell's
   value are unverified), and modes other than continuous (refused before any write).

Simulators: `tests/fakelogger.FakeL2` answers every write above as the loggers did. `FakeDuet` and a new
`FakeConcerto` (built on a real 1037-byte concerto header) use it. Memory follows the writes: `stop` appends an
event, `memclear` empties it, an enable writes a fresh header. Tests: `tests/test_l2_configure.py`, and the
review-6 duet tests, rewritten.

## Bench plan

Every step offloads first; nothing is erased before the logger's memory is saved and verified.

**Duets (A1):**
1. Read-only: incremental offload against the deployment already on disk, then `--full-download` to check that
   the incremental image equals a full read.
2. `--configure --period-ms 500` (enabled), log a few minutes, then offload twice. The first starts a new
   deployment (the erase); the second reads only the new bytes.
3. The final state, `--configure --period-ms 500 --no-enable`: cleared, clock set, 2 Hz, stopped. This is the
   first `memclear` on a duet. NOT READY (exit 2) is expected because of `--no-enable`.

**Concertos (A2):** the same, plus `sampling period = 167` (does the logger accept it?), and twist activation
on, enabled and gated.

**Concerto³ (B):** after phase B is built.

## Bench results, 2026-11-26 (duets SN082492, SN081500; concerto SN060275)

- **Incremental offloads work on both models**, from odd byte offsets. On one duet the result was byte-identical
  to a full read: `--full-download`, plus the prefix hash of the incremental record.
- **A clock write leaves the UTC offset `unknown`**, even straight after `settings offsetfromutc = 0.0`. Ruskin's
  loggers read +0.00 only because Ruskin rewrites every setting after the clock. The offset is now written after the
  clock set and checked in the readback.
- `enable erasememory = true` took 3.6 s on a duet holding 134 MB, so the enable waits up to 60 s for its reply.
  `memclear` answers `memclear used = 0`, takes 0.2-3.5 s, and leaves a duet `disabled`.
- **Concerto:** `sampling period = 250` and `twistactivation state = off|on` are answered with the command text.
  An erasing enable with twist activation on and the cap off answers `enable = gated`. The cap covers the USB
  port, so a concerto cannot be twisted on while connected.
- **Dropped commands:** 0 in 3,341 duet and 966 concerto commands, with every probe at n = 3. This is strong
  support for the pacing hypothesis, but not a controlled test: there was no unpaced run that day.
- A table read locked at one offload and unlocked at the next is not a change of the logger: only the channels both
  views show are compared.

- **A concerto at `period = 167` samples at exactly 6 Hz** (5.99985 Hz measured from memory growth over 590 s): a
  sub-second period is a rounded label for an exact rate, as the L3 reference says for Gen3. The L2 decoder timed
  sample n at n x 167 ms (2 ms/s late). It now uses k x 1000 / rate (`rawbin.sample_offsets_ms`).
- Both older concertos are at 167 ms (6 Hz), twist activation on, enabled, gated: Ruskin 2.26.2 shows 6 Hz as their
  maximum.

## Phase B (concerto³, Gen3): built 2026-11-26, simulator-tested, bench next

- `configure._apply_gen3`, Ruskin's sequence: `disable`; `clock datetime = X, offsetfromutc = 0.0`; the offset
  zeroed again if it does not read zero (a clock write erases it, L3 ref 4.1.1); `deployment endtime = ..,
  starttime = ..`; `sampling period = N` (sub-second only from the logger's `availablefastperiods`); `twistactivation
  enabled = true|false`; `verify`; `enable erasememory = true, memformat = CALBIN00`; or `permit command = memclear`
  + `memclear` without enable. Only what changes is written.
- `Gen3Driver.download` incremental: dataset 2 (header) read whole and compared; dataset 0 (events) read whole,
  must extend; dataset 1 head and tail blocks compared, then only new bytes.
- Tests: `tests/test_gen3_configure.py`; `FakeConcerto3` answers the writes as SN233442 answered Ruskin.

## Open questions

- Bench phase B on a concerto³ (read-only incremental first; the target is 31 ms, 32 Hz, twist on, gated).
