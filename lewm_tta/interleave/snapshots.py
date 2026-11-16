"""Stop-point plans, model versions and instrumentation-free prefix accounting."""

from copy import deepcopy
from dataclasses import dataclass
import time

import torch

from ..planning.cem import Plan
from ..rollout.support_updates import synchronize


@dataclass
class Snapshot:
    iterations: int
    plan: Plan
    model_version: int
    checkpoints: list
    adaptation: dict
    counters: dict
    timing_seconds: dict
    primary_seconds: float
    capture: dict


class SnapshotRecorder:
    """Instrumentation never restores weights or accesses query futures in search.

    One CPU copy of the adaptable parameters is retained per encountered model
    version. Copies are restored only after the entire search for diagnostics.
    Intermediate terminal-mean probes restore RNG and phase timer state. Their
    runtime is charged to their own stop point, then excluded from later ones.
    Actual instrumented elapsed time and extra work remain explicit in reports.
    """

    def __init__(self, session, started):
        self.session, self.started = session, started
        self.snapshots, self.states = {}, {}
        self.overhead_seconds = 0.
        self.extra_terminal_probes = 0

    def __call__(self, iterations, mean, trace, checkpoints, version, final_cost, signature):
        session, adapter = self.session, self.session.adapter
        synchronize(adapter)
        entered = time.perf_counter()
        overhead_before = self.overhead_seconds
        phase_before = dict(session.timings)
        terminal_probe_seconds = 0.
        if final_cost is None:
            devices = [adapter.device.index or 0] if adapter.device.type == "cuda" else []
            probe_started = time.perf_counter()
            try:
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    final_cost = float(session.score(mean[None])[0])
                synchronize(adapter)
                terminal_probe_seconds = time.perf_counter()-probe_started
                phase = dict(session.timings)
            finally:
                session.timings.clear()
                session.timings.update(phase_before)
            primary_seconds = entered-self.started-overhead_before+terminal_probe_seconds
            self.extra_terminal_probes += 1
        else:
            phase = phase_before
            primary_seconds = entered-self.started-overhead_before
        audit_started = time.perf_counter()
        adapted = session.adaptation_record()
        synchronize(adapter)
        audit_seconds = time.perf_counter()-audit_started
        primary_seconds += audit_seconds
        check_started = time.perf_counter()
        adapter.check_fixed_state()
        synchronize(adapter)
        check_seconds = time.perf_counter()-check_started
        primary_seconds += check_seconds
        if not torch.isfinite(torch.tensor(final_cost)):
            raise ValueError("Snapshot terminal-mean score must be finite")
        if version not in self.states:
            self.states[version] = [parameter.detach().cpu().clone() for parameter in adapter.parameters]
        adapted = deepcopy(adapted)
        if adapted["model_version"] != version:
            raise ValueError("Snapshot model version differs from adaptation state")
        plan = Plan(mean.detach().clone(), final_cost, deepcopy(trace),
                    session.config["planning"]["population"]*iterations+1, signature)
        snapshot = Snapshot(iterations, plan, version, deepcopy(checkpoints), adapted,
            dict(session.counters), phase, primary_seconds,
            {"mode": "single-pass", "model_version": version,
             "elapsed_measurement_overhead_before_stop": overhead_before,
             "terminal_probe_seconds": terminal_probe_seconds,
             "adaptation_audit_seconds": audit_seconds,
             "fixed_state_check_seconds": check_seconds,
             "latency_scope": "search prefix excluding earlier snapshot/probe overhead, including this stop's terminal-mean score, adaptation audit and fixed-state checks; snapshot copies and later diagnostics excluded"})
        self.snapshots[iterations] = snapshot
        synchronize(adapter)
        self.overhead_seconds += time.perf_counter()-entered

    def restore(self, snapshot):
        """Call only after the full CEM search, when optimizer state is unused."""
        adapter = self.session.adapter
        adapter.reset(self.session.reset_seed)
        with torch.no_grad():
            for parameter, value in zip(adapter.parameters, self.states[snapshot.model_version]):
                parameter.copy_(value.to(parameter.device))
        adapter.check_fixed_state()
