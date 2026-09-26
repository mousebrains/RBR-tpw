# RBR-tpw

Command-line offload and setup of RBR oceanographic loggers, without the Ruskin GUI.

Plug in a logger and `rbr-offload`:

1. measures the logger's clock skew against UTC to a few milliseconds,
2. downloads its memory, CRC-checked and resumable,
3. writes a CF-1.13 NetCDF file, with the serial number, clock skew, and memory and battery state as global
   attributes,
4. optionally sets the clock, resets the battery counter, erases memory, and re-enables logging with a new
   schedule,
5. tells you to unplug it, then waits for the next logger.

Several loggers can be plugged in at once; each is offloaded in parallel by its own worker.

## Supported loggers

| Logger | `id fwtype` | Offload (read-only) | NetCDF | Configure |
|---|---|---|---|---|
| RBRsolo T, firmware 1.000 | 9 | yes, tested on a logger | yes | yes |
| RBRsolo T, firmware 1.110 | 0 | yes | yes (same memory format as fwtype 9, checked against Ruskin on 34 files) | – |
| RBRduet (L2) | 102 | yes | yes, incl. pressure corrected with the compensation thermistor; matches Ruskin to ≤1.1e-13 on 7 files | – |
| RBRconcerto (L2) | 103 | yes | yes, incl. corrected conductivity and pressure; matches Ruskin to ≤1.1e-13 on 10 files | – |
| RBRconcerto³ and other Gen3 (L3) | 104 | yes | yes, EasyParse (`calbin00`) memory; matches Ruskin exactly on 10 files | – |
| Gen4 (L3.5) | 120 | from RBR's command reference only; untested | untested | – |
| Anything else | – | detected and skipped; nothing is changed | – | – |

Offload only reads from the logger. Whatever the decoder does, the raw memory, the settings record and the
serial transcript are always saved. `rbr-offload DIR --rebuild DIR/raw/<SN>_<time>.json` can then convert a
download later, e.g. after a decoder is added. `--configure` refuses any logger but fwtype 9: it erases memory,
and the write sequence has only been checked on that model. A Gen4 logger's download keeps every dataset
and schedule, but its NetCDF holds only the latest dataset's first schedule. A warning names any other
dataset or schedule that holds data.

The command sequences come from what RBR's Ruskin 2.26.1 sends each model (its serial logs), and from RBR's
command references. Only the fwtype-9 solo has been run against real hardware so far. The fwtype-9 protocol is
not in RBR's published references; it was worked out on a real logger and checked against Ruskin's own output.

Ruskin `.rsk` files from any RBR logger can be converted to the same NetCDF format; see
[Convert Ruskin .rsk files](#convert-ruskin-rsk-files).

## Install

```sh
pip install rbr-tpw          # or: uv tool install rbr-tpw
```

Requires Python ≥ 3.13, on macOS, Linux or Windows. The logger appears as a USB serial port (`/dev/cu.usbmodem*` on macOS, `/dev/ttyACM*` on Linux, `COM<n>` on Windows) and needs no
driver. **Quit Ruskin first**: it polls every RBR port it sees, and `rbr-offload` refuses to start while it is
running.

Loggers are recognized as Ruskin 2.26.1 recognizes them: USB vendor ID 0x0451, product ID 0xBEF0–0xBEFF. The
manufacturer string "RBR" is a second test. On Windows the logger uses Windows' own USB serial driver, and
Windows reports that driver's maker ("Microsoft") in place of the logger's, so the IDs are what identify it.
The USB details of each port go into the session log and the offload record. Windows support is tested in CI
with simulated loggers only so far.

## Offload

```sh
rbr-offload /path/to/data            # handle every logger plugged in, in parallel, until Ctrl-C
rbr-offload /path/to/data --once     # the logger(s) connected now (or the first to appear), then exit
                                     # (after a 3 s wait for any logger still being recognized)
```

Each logger gets its own worker, so four loggers on a hub download at the same time. Console lines are
tagged with the logger, e.g. `[SN100689@usbmodem101]`. Progress is shown every 10%, and "done with …:
disconnect the logger" tells you when a logger can be unplugged. Two steps depend on the computer's clock to
the millisecond: the clock-skew measurement and `--configure`'s clock set. They run one logger at a time,
which staggers the start of each download by about 4 s. Questions (`--configure`'s "configure SN…?" and
alarm acknowledgements) are asked one at a time, naming the logger. Other output waits until you answer.

Ctrl-C stops each download after its current block; reconnecting the logger resumes it. A logger that is
being configured finishes its configuration first, so it is never left erased but not logging. A second
Ctrl-C quits at once and names anything it interrupted. A port already opened by another copy of
`rbr-offload` is skipped.

So that a script can tell, `rbr-offload` exits with status bits set:

- 1 means an offload is incomplete: nothing was downloaded (an unsupported logger or an error), a download
  failed or was stopped, or its NetCDF was not written. A logger without a decoder yet doesn't count, since
  its saved download is all there is to get.
- 2 means a logger ended NOT READY TO DEPLOY. That covers a `--configure` that failed, was skipped (a model it
  doesn't support, or Ctrl-C) or was declined, and a logger that isn't logging afterwards.

A run where both happen exits with 3.

For each logger this writes:

| File | Contents |
|---|---|
| `SN_YYYYMMDDTHHMMSSZ.nc` | CF-1.13 NetCDF: `time`, one variable per channel (e.g. `temperature`, degree_Celsius), the raw readings, quality flags, the logger's own timestamps (`logger_time`), and the event list |
| `raw/SN_….bin` | the logger memory exactly as downloaded |
| `raw/SN_….json` | logger settings, calibration, clock skew, NTP offset, memory and battery state |
| `raw/SN_….log` | serial transcript: every command, reply, discarded byte, timeout and retry, with UTC ms timestamps. It is written as it happens, so it survives a failure, and it is named by port (`raw/<UTC>_usbmodem….log`) until the logger reports its serial number. |
| `raw/SN_…_configure.json` | with `--configure`: each step and the values read back |

Each run also writes `raw/rbr-offload_<UTC>.log`, the session log. It has every step of every logger, the
serial traffic, prompts and answers, and full error tracebacks, with UTC ms timestamps and the thread and
logger on each line. The console shows only the summary lines.

`rbr-offload DIR --rebuild DIR/raw/*.json` regenerates the NetCDF files from the raw files without the
logger, e.g. after a decoder fix. Configure reports (`*_configure.json`) are skipped. A record that can't be
converted is reported and the others are still done; the exit status is then 1.

## Convert Ruskin .rsk files

```sh
rbr-rsk2nc OUTDIR file.rsk ...          # or directories: searched recursively, sub-folders kept under OUTDIR
```

`rbr-rsk2nc` writes each `.rsk` as a NetCDF file in the same format, named after the `.rsk`. It never
modifies the `.rsk`. It skips files already converted, so an interrupted batch resumes (`--force`
overwrites). Each file takes one of two routes, recorded in the `conversion_route` attribute:

| Logger | Route | Contents |
|---|---|---|
| RBRsolo, fwtype 9 or 0 | `decode` | Ruskin keeps the logger's memory image in the `.rsk`. rbr-tpw decodes it exactly as `rbr-offload` does, raw readings included, and compares it with Ruskin's values (`ruskin_comparison`). |
| RBRduet, RBRconcerto, RBRconcerto³, … | `values` | Ruskin's computed values, in the same layout without the `_raw` variables. Channels Ruskin hides are included, without a CF `standard_name`. Channels the file lists without values are named in `ruskin_channels_not_stored`. |

The clock skew comes from Ruskin's `loggerTimeDrift`: logger minus the Ruskin computer's clock, whose offset
from UTC is unknown. It is in `clock_skew_vs_host_s`, and `clock_skew_s` is NaN. Samples on a clock that had
restarted at 2000-01-01 are re-timed with it, as `rbr-offload` does. EasyParse files (RBRconcerto³) have no
drift recorded. `--ruskin-values` uses Ruskin's values for the solos too.

Offloading never changes anything on the logger. A logger that was logging keeps logging.

### Clock skew

The logger reports whole seconds, so the tool polls its clock until the second ticks. It brackets the tick
between the send and receive times and repeats this 3 times. The computer's clock is referenced to UTC by an
SNTP query made from Python (`--ntp-server`, default `time.apple.com`; `--no-ntp` to skip). It uses the
lowest-delay of 4 replies and is reused for 5 minutes. The reported uncertainty is typically about ±9 ms for
the tick measurement plus the NTP uncertainty: half the round-trip delay plus the server's root delay/2 and
root dispersion, about ±12 ms over the internet.

If the logger lost power during a deployment, its clock restarts at 2000-01-01. Samples taken after the last
such reset are re-timed with the skew measured at offload and flagged in `time_flag`. This includes a logger
enabled after its clock had already reset. Samples between two resets have no recoverable time. They are left
out of the NetCDF with a warning and kept in the raw file. Only a clock restart (an RTC-reset event) splits a
record this way; ordinary time anchors, such as an RBRconcerto's twist-activation events, do not.

## Alarms: battery and remaining sampling time

At every offload, and again after `--configure`, the tool estimates how long the logger can keep
sampling on its current schedule:

- **Memory-limited days** are exact: free bytes ÷ bytes per day.
- **Energy-limited days** come from a model: the logger's energy counter derated to 90%, minus the modelled
  energy for the samples already in memory, divided by the modelled use per day. The model uses RBR Ruskin's
  constants for the solo T (0.69 mA while sampling, 5.5 µA asleep, 3.6 V).

The lesser of the two is reported and stored in the NetCDF attributes. You get a loud alarm (a bell, a red
banner, and a prompt to acknowledge unless `--yes` is given) when:

- the battery reads below `--min-voltage` (default 3.3 V), or
- fewer than `--min-days` days of sampling remain, or
- a configured end time comes after the estimated run-out time.

The battery voltage is only a coarse check. A lithium thionyl chloride (Li-SOCl2) cell stays near 3.6 V for
most of its life, so a low reading means a dead cell, a bad contact, or a cell close to exhaustion. The
energy estimate is the better measure of capacity, but it is a model: check it against Ruskin before relying
on it.

## Configure and enable

```sh
rbr-offload DIR --configure deploy.yaml             # settings from a file
rbr-offload DIR --configure --period-ms 1000 \
            --start 2026-10-01T00:00:00Z --end never --fresh-battery
rbr-offload DIR --configure deploy.yaml --used-battery 14/50   # a cell with 14 of ~50 days used elsewhere
```

`--configure` runs only after that logger's data has been downloaded and saved. It shows the plan and asks
before changing each logger (`--yes` skips the question). It then:

1. stops logging,
2. sets the clock to UTC, aligned to the second and checked afterwards,
3. optionally resets the battery energy counter: `--fresh-battery` when a new cell was installed, or
   `--used-battery USED/LIFE` for a cell already used elsewhere, e.g. `14/50` for 14 days used of an expected
   50-day life in that instrument. The counter is set to (LIFE − USED)/LIFE of a new cell, so the energy
   estimate starts from that fraction of the derated capacity. This assumes the cell was drained at a steady
   rate there; a Li-SOCl2 cell's voltage is too flat to tell how much it has left.
4. writes the schedule,
5. erases memory,
6. enables logging,
7. reads everything back.

If any step fails, or the logger isn't logging (or pending) afterwards, you get a loud alarm saying whether
memory was erased. If the reply to the erase was lost, it says the memory may have been erased. The final line then reads "NOT READY TO DEPLOY" instead of just "disconnect".
`--no-erase` is refused when memory holds data and logging would be enabled, because the logger itself
refuses that; use it with `--no-enable` to change settings only. A logger is configured at most once per run,
so unplugging and replugging it won't erase it again. The exception is a logger whose configure failed:
reconnecting it tries again. The offload that runs first saves whatever that logger holds.

Settings can live in a YAML file (`--config settings.yaml`, or `--configure settings.yaml`); command-line
options override it. See [deploy.example.yaml](https://github.com/mousebrains/RBR-tpw/blob/main/deploy.example.yaml):

```yaml
thresholds:
  min_battery_voltage: 3.3
  min_days: 90
deploy:                  # used only with --configure
  set_clock: true
  fresh_battery: false
  # battery_days_used: 14   # a used cell: 14 days used of an expected
  # battery_life_days: 50   # 50-day life (instead of fresh_battery)
  erase: true
  enable: true
  schedule:
    mode: continuous     # only continuous so far
    period_ms: 500       # 500 (2 Hz) or whole seconds; or rate_hz
    start: now           # now | ISO-8601 UTC
    end: never           # never | ISO-8601 UTC
```

Other options: `--no-erase`, `--no-enable`, `--no-clock`, `--rate-hz`.

## Development

```sh
uv sync
uv run pytest          # decoder checked against Ruskin output; NetCDF checked with the IOOS compliance-checker
```

Tests that need recorded logger data are skipped when it is absent: `tests/data/` and `RBR_TPW_RSK_DIR`, a
folder of Ruskin `.rsk` files. Neither is in this repository.

The tests run offloads against simulated loggers (`tests/fakelogger.py`): an RBRsolo (fwtype 9 or 0), an RBRduet
and an RBRconcerto³. Their replies copy what real loggers sent, and their memory uses the real formats with
real CRCs. They are reached three ways:

- in-process, by replacing `rbr_tpw.link.open_serial`, for protocol, decoding, threading and Ctrl-C tests;
- over a local TCP socket (`socket://127.0.0.1:<port>`, every OS), through real pyserial;
- over a pseudo-terminal (macOS and Linux), which pyserial opens like a USB serial port, with port setup
  and the exclusive lock.

`rbr-offload --port socket://host:port` also works, e.g. to try the tool against a simulated logger.

## Disclaimer

Not affiliated with or endorsed by RBR Ltd. RBR, RBRsolo and Ruskin are RBR Ltd trademarks. Commands that
write to a logger (`--configure`) erase its memory; keep the raw files.

## License

GPL-3.0-or-later
