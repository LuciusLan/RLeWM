"""Terminal progress for current evaluators and unchanged historical replays."""

import os
import re
import signal
import subprocess
import sys

from tqdm import tqdm


def enabled(config):
    """Retain log_every=0 as the existing quiet setting."""
    quiet = os.environ.get("TQDM_DISABLE", "").lower() in ("1", "true", "yes")
    return not quiet and config.get("evaluation", {}).get("log_every", 1) != 0


def progress(iterable=None, *, config, desc, unit="query", show=True, **kwargs):
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True,
                mininterval=1, disable=None if show and enabled(config) else True,
                **kwargs)


_QUERIES = re.compile(r"^(?P<label>.+?): (?P<done>[\d,]+)/(?P<total>[\d,]+) (?:queries|contexts)\b")
_TEMPORAL = re.compile(r"^context (?P<query>\d+)/(?P<total>\d+) seed (?P<seed>\d+) (?P<mode>[\w-]+): (?P<phase>.+)")
_CANDIDATES = re.compile(r"^\s+candidate-temporal: (?P<done>\d+)/(?P<total>\d+) sequences in population; (?P<updates>\d+) cumulative updates")


class _ArchivedProgress:
    """Render known progress messages without modifying a locked child process."""

    def __init__(self, config):
        self.config = config
        self.queries = self.candidates = None
        self.temporal = False

    def consume(self, line):
        if self.temporal and line.startswith("P3 temporal |"):
            self.close(success=True)
            return False
        match = _QUERIES.match(line)
        if match:
            done, total = (int(match[name].replace(",", "")) for name in ("done", "total"))
            if self.queries is None:
                # The archive's first message may arrive after eight queries.
                # Treat that prefix as completed work, not an instantaneous rate.
                self.queries = progress(total=total, initial=done, config=self.config,
                                        desc=match["label"])
            self.queries.update(done - self.queries.n)
            return True
        match = _TEMPORAL.match(line)
        if match:
            self.temporal = True
            if self.queries is None:
                self.queries = progress(total=int(match["total"]), config=self.config,
                                        desc="P3 temporal")
            # A new context proves that the previous context's diagnostics ended.
            self.queries.update(int(match["query"]) - 1 - self.queries.n)
            if match["phase"].startswith("starting;"):
                self.queries.set_postfix(seed=match["seed"], mode=match["mode"])
            return True
        match = _CANDIDATES.match(line)
        if match:
            done, total = (int(match[name]) for name in ("done", "total"))
            if self.candidates is None:
                self.candidates = progress(total=total, config=self.config, desc="Candidate-temporal population",
                                           unit="sequence", position=1, leave=False)
            self.candidates.set_postfix(updates=match["updates"], refresh=False)
            self.candidates.update(done - self.candidates.n)
            if done == total:
                self.candidates.close()
                self.candidates = None
            return True
        return False

    def close(self, success):
        if self.candidates is not None:
            self.candidates.close()
        if self.queries is not None:
            if success and self.temporal:
                self.queries.update(self.queries.total - self.queries.n)
            self.queries.close()


def replay_with_progress(command, source):
    """Keep exact archived execution and hashes; translate only its console output."""
    config = source.get("config", {})
    if not sys.stderr.isatty() or not enabled(config):
        os.execv(sys.executable, command)
    renderer = _ArchivedProgress(config)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
    success = False
    try:
        for line in child.stdout:
            if not renderer.consume(line):
                tqdm.write(line, end="", file=sys.stdout)
        returncode = child.wait()
        success = returncode == 0
    except KeyboardInterrupt:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        child.stdout.close()
        renderer.close(success)
    if returncode:
        raise SystemExit(returncode if returncode > 0 else 128 - returncode)
