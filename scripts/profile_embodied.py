"""Bounded, headless speed measurement without saving learned weights."""
import argparse
import cProfile
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')

    import torch
    import fly_embodied as app
    from flygym import Fly

    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; refusing to report CPU fallback as GPU performance')
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    if args.reference:
        app.Fly = Fly
        original_init = app.BrainEngine.__init__

        def reference_init(self, *a, **kw):
            kw['propagation'] = 'sparse'
            original_init(self, *a, **kw)
        app.BrainEngine.__init__ = reference_init

    app.BrainEngine.save_plastic_weights = lambda self: print('[Profile] Weight saving disabled')
    original_step = app.HybridTurningController.step
    timings = {'steps': 0}
    profiler = cProfile.Profile() if args.profile else None

    def step(self, *a, **kw):
        if 'start' not in timings:
            torch.cuda.synchronize()
            timings['start'] = time.perf_counter()
            timings['dt'] = self.timestep
            if profiler:
                profiler.enable()
        result = original_step(self, *a, **kw)
        timings['steps'] += 1
        return result

    app.HybridTurningController.step = step
    sys.argv = ['fly_embodied.py', '--no-viewer', '--duration', str(args.seconds),
                '--visual', '--flight', '--olfactory', '--gustatory', '--somatosensory']
    started = time.perf_counter()
    app.main()
    torch.cuda.synchronize()
    ended = time.perf_counter()
    if profiler:
        profiler.disable()
        profiler.dump_stats(args.output + '.prof')
    elapsed = ended - timings['start']
    result = dict(reference=args.reference, profiled=args.profile,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  steps=timings['steps'], simulated_seconds=timings['steps'] * timings['dt'],
                  loop_wall_seconds=elapsed, total_wall_seconds=ended - started,
                  body_steps_per_second=timings['steps'] / elapsed,
                  realtime_factor=timings['steps'] * timings['dt'] / elapsed)
    Path(args.output + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
