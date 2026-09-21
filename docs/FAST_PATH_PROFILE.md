# Fast-path verification and speed profile, 2026-09-21

Reviewed revision `ea9792c` and the fast-path changes since `4c4e568`.
The current fast path completed the bounded GPU run and passed the checks below.
Measured throughput improved by 23.0% against the reference configuration,
but remains 0.055× real time with the tested sensory workload.

## Measurements

Each run advances 10,000 body steps, or 1.0 simulated second at a 0.0001 s
body timestep. The workload enables visual, flight, olfactory, gustatory, and
somatosensory processing. Viewer and monitor are disabled. The default headless
P9 stimulus is used. Existing learned weights are loaded but never saved by
the profiling wrapper. NumPy and PyTorch are seeded with 42.

| Configuration | Loop wall time | Body steps/s | Sim seconds/wall second | Startup + loop |
| --- | ---: | ---: | ---: | ---: |
| FastFly + event/fused brain | 18.099 s | 552.5 | 0.05525 | 24.601 s |
| Upstream Fly + sparse brain | 22.259 s | 449.3 | 0.04493 | 28.310 s |
| Fast path under cProfile | 20.660 s | 484.0 | 0.04840 | 27.321 s |

This is a 23.0% throughput increase, or an 18.7% reduction in loop wall time.
Use the unprofiled rows for speed comparisons. cProfile adds about 14% wall
time here. This is one sequential run per configuration, not a statistical
benchmark. An earlier fast-path run overlapped the correctness tests and is
excluded from the table. Other system workloads were not controlled.

The reference configuration switches both FastFly to upstream FlyGym and brain
propagation to sparse in the current checkout; it is not a historical checkout
and does not isolate the contribution of each optimization. Closed-loop
trajectories need not remain identical because event accumulation uses atomics.

Environment: shared `../fly-brain/.venv`, Python 3.10,
PyTorch `2.10.0+rocm7.1`, FlyGym 1.0.1, MuJoCo 3.2.2,
dm-control 1.0.22, EGL rendering. PyTorch identifies GPU 0 as
`AMD Radeon Graphics`, architecture `gfx1100`, 24,560 MB VRAM, PCI bus 3.
The simulated connectome has 138,639 neurons and 15,091,983 synapses.

## Where time goes

The detailed fast-path profile contains about 9.87 million calls.

| Operation | Calls | Time | Interpretation |
| --- | ---: | ---: | --- |
| HybridTurningController.step | 10,000 | 19.256 s cumulative | About 93% of loop wall time |
| FastFly.get_observation | 20,000 | 9.031 s cumulative | Twice per body step; includes physics refresh |
| MuJoCo mj_forward | 10,000 | 6.077 s self | Full refresh from dirty observation state |
| MuJoCo mj_step1 | 10,000 | 4.169 s self | Native physics work |
| MuJoCo mj_step2 | 10,000 | 1.892 s self | Native physics work |
| BrainBodyBridge.compute_drive | 9,999 | 0.612 s cumulative | Body-drive decoding |
| Tensor.item | 2,575 | 0.076 s self | Includes GPU synchronization/readouts |
| BrainEngine.step | 99 | 0.068 s cumulative | CPU-side time; GPU work is asynchronous |

Cumulative rows overlap and must not be added. The three native MuJoCo calls
have disjoint self times totaling 12.138 s, about 59% of measured loop time.
This points to physics and observation/controller work as the next optimization
target. Removing a dirty-state refresh without proving equivalent sensor/contact
state would risk changing behavior; this report makes no such code change.

The application advances one brain step per 100 body steps. Thus a one-second
body run performs only 100 neural steps, or 10 ms of neural model time. This ratio
predates the reviewed changes. The profile begins at the first controller step,
after the first neural update and eye render, so it records 99 brain steps and
18 eye renders. Timing stops after application cleanup and GPU synchronization;
it excludes imports, initialization, and the first pre-controller work.
It is not a GPU-kernel timing trace or an interactive viewer FPS measurement.

## Verification and remaining gaps

- `MUJOCO_GL=egl ../fly-brain/.venv/bin/python -m pytest tests -q`:
  **8 passed**. Includes 600-step upstream/FastFly trajectory comparisons with
  adhesion on/off, reset/dirty-state checks, shared-physics observations, and
  CPU readout/plasticity equivalence tests.
- Additional synthetic GPU checks passed: event propagation at four spike
  densities with batch size two and live weight mutation at tolerance 2e-5,
  100 fused neuron updates exactly equal to the reference, and an exact
  Hebbian update on 1,027 edges. These were ad hoc checks, not existing tests.
- Both full sensory benchmark configurations completed 10,000 body steps.
  No long-duration behavioral or numerical-equivalence claim is made.
- Both application loops now use
  `get_spike_readout()` to transfer DN and population readings together after
  each brain step. Neural timing and the decoder's `None` input when no
  populations are registered are preserved. The measurements above predate
  this change; no additional speedup has been measured. Deferred stepping and
  batching across multiple steps remain unused; `run_block()` runs one step.
- `save_plastic_weights()` does not finish queued step debt despite the
  new helper's stated saving contract. This matters if deferred stepping is
  integrated later; current application loops do not queue debt.
- Benchmark processes exited successfully but emitted `EGL_NOT_INITIALIZED`
  destructor warnings after writing results. Rendering initialization
  required EGL for the tests; the initial default-GL run failed four tests at
  context creation, before exercising their assertions.
- Viewer/monitor performance and startup stability were not reverified here.

## Reproduce without changing learned weights

### Exact readout comparison after integration

`scripts/check_spike_readout.py` compares both readout implementations on each
step of the same full-connectome event-backend GPU trajectory. This keeps
atomic accumulation and random draws identical for both readouts.
It runs 1,200 neural steps, or 120 ms neural time, cycling P9, sugar, LC4, JO,
bitter, and Or56a stimuli in 200-step blocks, with eight application populations.

The initial strict test found a one-float32-step difference at step 649:
JO_touch_L was 0.04054053872823715 using the CPU mean versus
0.04054054245352745 using the original GPU mean. The combined path now computes
population means on the GPU using the original operation, then packs those means
with DN spikes for a single CPU transfer.

The corrected code passed with zero exact readout, decoder-rate, drive, or
behavior-mode mismatches across all 1,200 steps. The run produced 1,492 total
neuron spikes and 17 DN spikes. Plasticity changed weights during the run;
readouts did not increment state/weight/input/accumulator tensor mutation
versions or consume GPU random numbers. No queued readouts or step debt remained.
The three CPU readout tests also passed. Saved weights were not written.

This isolates equivalence of the readout change; it does not compare the event
backend against sparse propagation or prove long-duration closed-loop behavior.

```bash
cd /home/stuffie/IDE/fly-brain-pristine
../fly-brain/.venv/bin/python scripts/check_spike_readout.py
```

### Speed measurements

Run sequentially from the repository root:

```bash
cd /home/stuffie/IDE/fly-brain-pristine
MUJOCO_GL=egl ../fly-brain/.venv/bin/python scripts/profile_embodied.py --seconds 1 --output /tmp/fly-fast
MUJOCO_GL=egl ../fly-brain/.venv/bin/python scripts/profile_embodied.py --seconds 1 --reference --output /tmp/fly-reference
MUJOCO_GL=egl ../fly-brain/.venv/bin/python scripts/profile_embodied.py --seconds 1 --profile --output /tmp/fly-profile
```

The wrapper refuses CPU fallback, disables saving only in its own process, and
writes JSON results plus a `.prof` file when requested. Increase `--seconds`
for a longer measurement. Raw results and profile summaries from this run are
in [profiling-2026-09-21](profiling-2026-09-21/).

## Normal application commands

Full interactive single fly. Close the viewer or press Ctrl+C to stop:

```bash
cd /home/stuffie/IDE/fly-brain-pristine
../fly-brain/.venv/bin/python fly_embodied.py --visual --monitor --flight --olfactory --gustatory --somatosensory
```

Bounded headless single fly:

```bash
MUJOCO_GL=egl ../fly-brain/.venv/bin/python fly_embodied.py --no-viewer --duration 1 --visual --flight --olfactory --gustatory --somatosensory
```

Normal runs save learned weights on exit. `--duration` is enforced only without
the viewer. FastFly is used automatically; eligible ROCm GPU weights select
event propagation automatically. Look for `Recurrent propagation: event`.
