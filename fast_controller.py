"""Reduce scalar interpolation overhead without changing gait or correction rules."""

from bisect import bisect_left
import math

import numpy as np
from flygym.examples.locomotion.turning_controller import HybridTurningController
from flygym.simulation import SingleFlySimulation


class ScalarLinearInterpolation:
    """Scalar path for the controller's linear, extrapolating SciPy interpolators."""

    def __init__(self, reference):
        self.reference = reference
        self.x = tuple(float(x) for x in reference.x)
        self.y = tuple(float(y) for y in reference.y)
        self.slopes = tuple((self.y[i+1] - self.y[i]) / (self.x[i+1] - self.x[i])
                            for i in range(len(self.x) - 1))

    def __call__(self, value):
        if not isinstance(value, (float, int, np.floating)) or not math.isfinite(value):
            return self.reference(value)
        value = float(value)
        high = min(max(bisect_left(self.x, value), 1), len(self.x) - 1)
        low = high - 1
        return self.slopes[low] * (value - self.x[low]) + self.y[low]


class _ObservationCache(SingleFlySimulation):
    """Supply the previous post-step observation to the gait controller."""

    def get_observation(self):
        cached = self._cached_body_observation
        if (self.reuse_observations and cached is not None
                and not self.physics.is_dirty):
            return cached
        return super().get_observation()


class FastTurningController(HybridTurningController, _ObservationCache):
    def __init__(self, *args, reuse_observations=True, **kwargs):
        self.reuse_observations = reuse_observations
        self._cached_body_observation = None
        super().__init__(*args, **kwargs)

    def invalidate_observation(self):
        """Call after physics changes outside the controller's own step."""
        self._cached_body_observation = None

    def reset(self, *args, **kwargs):
        self.invalidate_observation()
        obs, info = super().reset(*args, **kwargs)
        self._cached_body_observation = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._cached_body_observation = obs
        return obs, reward, terminated, truncated, info

    def _init_phasic_gain(self, swing_extension=np.pi / 4):
        gains = super()._init_phasic_gain(swing_extension)
        return {leg: ScalarLinearInterpolation(gain) for leg, gain in gains.items()}
