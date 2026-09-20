"""Inference-only AlphaLIF update with the reference float32 operation order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _step(poisson, recurrent, conductance, delay, spikes, voltage, refractory,
          result, next_delay, N: tl.constexpr, TOTAL: tl.constexpr,
          DELAY: tl.constexpr, SYN_DECAY: tl.constexpr, MEM_FACTOR: tl.constexpr,
          REST: tl.constexpr, RESET: tl.constexpr, THRESHOLD: tl.constexpr,
          REFRAC: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    batch, neuron = i // N, i % N
    g = tl.load(conductance + i, valid, other=0)
    old_spike = tl.load(spikes + i, valid, other=0)
    v = tl.load(voltage + i, valid, other=0)
    ref = tl.load(refractory + i, valid, other=0)
    ref = ref * (1.0 - old_spike)
    ref = ref + 1.0
    delayed = tl.load(delay + batch * DELAY * N + neuron, valid, other=0)
    next_g = g * SYN_DECAY + delayed * (ref > REFRAC).to(tl.float32)
    next_v = v + MEM_FACTOR * (g - (v - REST))
    spike = (next_v - THRESHOLD > 0).to(tl.float32)
    next_v = next_v - (next_v - RESET) * spike
    next_g = next_g - next_g * spike
    tl.store(result + i, next_g, valid)
    tl.store(result + TOTAL + i, spike, valid)
    tl.store(result + 2 * TOTAL + i, next_v, valid)
    tl.store(result + 3 * TOTAL + i, ref, valid)
    incoming = SCALE * (tl.load(poisson + i, valid, other=0)
                        + tl.load(recurrent + i, valid, other=0))
    tl.store(next_delay + (batch * DELAY + DELAY - 1) * N + neuron, incoming, valid)


class FusedNeuronStep:
    def __init__(self, neurons, scale):
        self.neurons = neurons
        self.scale = scale

    def __call__(self, poisson, recurrent, conductance, delay, spikes, voltage, refractory):
        # Each returned state owns new storage, just like the reference path.
        # Preserve the public delay-buffer ordering and keep snapshots valid.
        next_delay = torch.roll(delay, shifts=-1, dims=1)
        result = torch.empty((4, *conductance.shape), device=conductance.device,
                             dtype=conductance.dtype)
        n = self.neurons
        _step[(triton.cdiv(conductance.numel(), 256),)](
            poisson, recurrent, conductance, delay, spikes, voltage, refractory,
            result, next_delay, conductance.shape[1], conductance.numel(), delay.shape[1],
            1 - n.synapse.time_factor, n.neuron.time_factor, n.neuron.v_rest,
            n.neuron.v_reset, n.neuron.v_threshold, n.steps_refrac, self.scale,
            BLOCK=256, enable_fp_fusion=False)
        return result[0], next_delay, result[1], result[2], result[3]
