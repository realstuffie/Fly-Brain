"""Measure reuse of the last body observation in the walking controller.

The controller reads the pre-step observation. FlyGym also reads the post-step
observation, which is the same body state at the start of the next step unless
other code changes physics between steps. This benchmark checks both speed and
state equivalence for an uninterrupted walking controller run.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmark_native_mujoco import make_simulation, snapshot


def run(sim, steps, warmup, reuse):
    sim.reuse_observations = reuse
    sim.reset(seed=0)
    checks = {}
    start = None
    for step in range(steps):
        if step == warmup:
            start = time.perf_counter()
        drive = np.array(((1.0, 1.0), (0.6, 1.2), (0.0, 0.0))[
            min(3 * step // steps, 2)])
        observation, _, _, _, info = sim.step(drive)
        if step in (warmup - 1, steps // 2, steps - 1):
            checks[step] = snapshot(sim, observation, info)
    return time.perf_counter() - start, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.warmup < args.steps:
        parser.error("require 0 < warmup < steps")

    sim = make_simulation()
    try:
        reference_seconds, reference = run(sim, args.steps, args.warmup, False)
        reuse_seconds, candidate = run(sim, args.steps, args.warmup, True)
    finally:
        sim.close()

    errors = {
        f"{step}:{name}": float(np.max(np.abs(expected - candidate[step][name])))
        for step, fields in reference.items()
        for name, expected in fields.items()
    }
    result = {
        "steps": args.steps,
        "warmup": args.warmup,
        "reference_seconds": reference_seconds,
        "reuse_seconds": reuse_seconds,
        "speedup": reference_seconds / reuse_seconds,
        "max_error": max(errors.values()),
        "errors": errors,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["max_error"] > 1e-5:
        raise SystemExit("Observation reuse changed the trajectory")


if __name__ == "__main__":
    main()
