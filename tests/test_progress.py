# ruff: noqa: F811  (the `rig` fixture is imported from test_offload and named by every test)
"""The progress meter: a status line on a terminal (Console.status), fed by the workers' stages; nothing off a
terminal (PLAN-incremental-offload.md, tests 17-19)."""

import io
import threading

from test_offload import fast_skew, rig  # noqa: F401  (a fixture)

from rbr_tpw import cli
from rbr_tpw.console import Console, setup_logging

CLEAR = "\r\033[2K"


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_status_line_is_drawn_in_place_and_cleared_around_output(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback=(80, 24): type("S", (), {"columns": 21})())
    out = Terminal()
    console = Console(stream=out, input_fn=lambda q: "y", interactive=True)
    assert console.status_enabled
    console.status("downloading 31%")
    assert out.getvalue() == CLEAR + "downloading 31%"
    console.write("a log line")
    assert out.getvalue() == CLEAR + "downloading 31%" + CLEAR + "a log line\n" + "downloading 31%"
    out.seek(0), out.truncate()
    assert console.ask("go? ") == "y"  # the prompt never lands on the status line, which comes back after it
    assert out.getvalue().startswith(CLEAR) and out.getvalue().endswith("downloading 31%")
    out.seek(0), out.truncate()
    console.status("x" * 40)  # cut to the terminal width
    assert out.getvalue() == CLEAR + "x" * 20
    console.status(None)
    assert out.getvalue().endswith(CLEAR)
    console.write("after")
    assert out.getvalue().endswith(CLEAR + "after\n")  # no status line to redraw


def test_no_status_line_off_a_terminal(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    rig.add("usbmodem1", serial=21, n_samples=40_000)
    cli.run(rig.settings(), once=True, port=None)
    text = rig.console_text()
    assert "\r" not in text and "downloaded 0.16 of 0.16 MB (100%)" in text  # the 10% lines, as before


def test_status_line_on_a_terminal_replaces_the_progress_lines(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    out = Terminal()
    console = Console(stream=out, input_fn=lambda q: "", interactive=False)
    session_log = rig.tmp / "raw" / "tty-session.log"
    setup_logging(console, session_log)
    rig.add("usbmodem1", serial=22, n_samples=40_000, bytes_per_s=400_000)
    s = rig.settings()
    s.console = console
    cli.run(s, once=True, port=None)
    text = out.getvalue()
    assert CLEAR + "[SN22@usbmodem1] downloading " in text and "MB [" in text and "% " in text
    assert "downloaded 0.16 of 0.16 MB" not in text  # on a terminal the bar says it
    assert "downloaded 0.16 of 0.16 MB (100%)" in session_log.read_text()  # the session log keeps it
    # the worker's last stage is cleared before "done with" is printed, so nothing is left on the line
    assert text.endswith("disconnect the logger\n") and CLEAR + "[SN22@usbmodem1] done with" in text
    assert cli._status_console is None
    setup_logging(Console(stream=io.StringIO()))


def test_two_loggers_on_a_terminal_show_the_summary_not_a_bar(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    monkeypatch.setattr(cli, "STATUS_EVERY_S", 0.05)
    out = Terminal()
    console = Console(stream=out, input_fn=lambda q: "", interactive=False)
    session_log = rig.tmp / "raw" / "tty-session2.log"
    setup_logging(console, session_log)
    for i in (1, 2):
        rig.add(f"usbmodem{i}", serial=30 + i, n_samples=60_000, bytes_per_s=300_000)
    s = rig.settings()
    s.console = console
    cli.run(s, once=True, port=None)
    text = out.getvalue()
    assert CLEAR + "in progress: SN31@usbmodem1 " in text
    assert "in progress:" not in session_log.read_text()  # the 30 s summary line is not logged on a terminal
    setup_logging(Console(stream=io.StringIO()))
