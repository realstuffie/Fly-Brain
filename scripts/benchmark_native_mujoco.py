"""Compare FlyGym's body step with a direct MuJoCo step on the same fly.

This keeps the gait controller and FastFly observations in both paths. It
measures the potential gain from replacing only FlyGym's inner step wrapper.
Run with MUJOCO_GL=egl when no desktop display is available.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from flygym.simulation import SingleFlySimulation

from fast_controller import FastTurningController
from fast_fly import FastFly
from looming_arena import LoomingArena
from physics_refresh import reuse_physics_stages


def make_simulation():
    contacts = [
        f"{leg}{segment}"
        for leg in ("LF", "LM", "LH", "RF", "RM", "RH")
        for segment in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
    ]
    fly = FastFly(enable_adhesion=True, draw_adhesion=False,
                  contact_sensor_placements=contacts, enable_vision=False)
    sim = FastTurningController(fly=fly, arena=LoomingArena(),
                                timestep=1e-4, seed=0)
    reuse_physics_stages(sim.physics)
    return sim


def native_step(self, action):
    """Direct replacement for SingleFlySimulation.step, for this benchmark."""
    fly = self.fly
    self.arena.step(dt=self.timestep, physics=self.physics)
    data = self.physics.data.ptr
    data.ctrl[self._benchmark_joint_ids] = action["joints"]
    data.ctrl[self._benchmark_adhesion_ids] = action["adhesion"]
    fly._last_adhesion = action["adhesion"]
    model = self.physics.model.ptr
    # dm_control's legacy Euler step integrates first, then refreshes the new
    # position and velocity state. Plain mj_step leaves a different readout.
    mujoco.mj_step2(model, data)
    mujoco.mj_step1(model, data)
    # Match the current physics_refresh path before reading acceleration sensors.
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_VEL, 0)
    self.physics._dirty = False
    self.curr_time += self.timestep
    return fly.post_step(self)


def snapshot(sim, observation, info):
    data = sim.physics.data.ptr
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "sensordata": data.sensordata.copy(),
        "position": observation["fly"][0].copy(),
        "joints": observation["joints"].copy(),
        "contact_forces": observation["contact_forces"].copy(),
        "end_effectors": observation["end_effectors"].copy(),
        "action_joints": info["joints"].copy(),
        "action_adhesion": info["adhesion"].copy(),
    }


def run(sim, steps, warmup):
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.warmup < args.steps:
        parser.error("require 0 < warmup < steps")

    sim = make_simulation()
    original_step = SingleFlySimulation.step
    try:
        reference_seconds, reference = run(sim, args.steps, args.warmup)
        model = sim.physics.model
        sim._benchmark_joint_ids = np.array([
            model.name2id(actuator.full_identifier, "actuator")
            for actuator in sim.fly.actuators], dtype=np.intp)
        sim._benchmark_adhesion_ids = np.array([
            model.name2id(actuator.full_identifier, "actuator")
            for actuator in sim.fly.adhesion_actuators], dtype=np.intp)
        SingleFlySimulation.step = native_step
        native_seconds, candidate = run(sim, args.steps, args.warmup)
    finally:
        SingleFlySimulation.step = original_step
        sim.close()

    errors = {}
    for step, fields in reference.items():
        for name, expected in fields.items():
            actual = candidate[step][name]
            errors[f"{step}:{name}"] = float(np.max(np.abs(expected - actual)))
    max_error = max(errors.values())
    result = {
        "steps": args.steps,
        "warmup": args.warmup,
        "reference_seconds": reference_seconds,
        "native_seconds": native_seconds,
        "speedup": reference_seconds / native_seconds,
        "max_error": max_error,
        "errors": errors,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if max_error > 1e-5:
        raise SystemExit("Native step diverged from FlyGym; speedup is not usable yet")


if __name__ == "__main__":
    main()
