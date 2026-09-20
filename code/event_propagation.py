"""GPU propagation through the outgoing connections of active neurons."""

import numpy as np
import torch
import triton
import triton.language as tl
from scipy.sparse import csr_matrix


@triton.jit
def _compact_spikes(spikes, active, counts, N: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(1)
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    firing = tl.load(spikes + batch * N + indices, indices < N, other=0) != 0
    ranks = tl.cumsum(firing.to(tl.int32)) - 1
    count = tl.sum(firing.to(tl.int32))
    if count > 0:
        start = tl.atomic_add(counts + batch, count, sem='relaxed')
        tl.store(active + batch * N + start + ranks, indices, firing)


@triton.jit
def _propagate(spikes, active, counts, pointers, targets, edge_ids, weights, output,
               N: tl.constexpr, WORKERS: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(1)
    count = tl.load(counts + batch)
    lane = tl.arange(0, BLOCK)
    for index in range(tl.program_id(0), count, WORKERS):
        source = tl.load(active + batch * N + index)
        spike = tl.load(spikes + batch * N + source)
        start = tl.load(pointers + source)
        end = tl.load(pointers + source + 1)
        for offset in range(start, end, BLOCK):
            edge = offset + lane
            valid = edge < end
            target = tl.load(targets + edge, valid, other=0)
            edge_id = tl.load(edge_ids + edge, valid, other=0)
            value = tl.load(weights + edge_id, valid, other=0)
            tl.atomic_add(output + batch * N + target, value * spike,
                          valid, sem='relaxed')


class EventPropagation:
    """Keep a CSC edge index while reading live, mutable CSR weight values.

    Floating-point atomic addition can change summation order. The topology
    must remain fixed; in-place plasticity updates need no index rebuild.
    """

    def __init__(self, weights):
        if (weights.layout != torch.sparse_csr or weights.dtype != torch.float32
                or weights.device.type != 'cuda' or weights.shape[0] != weights.shape[1]):
            raise ValueError('Event propagation requires a square float32 GPU CSR matrix')
        self.size = weights.shape[0]
        if max(self.size, weights._nnz()) >= np.iinfo(np.int32).max:
            raise ValueError('Event propagation requires 32-bit edge and neuron indices')
        self.device = weights.device
        self.values = weights.values()
        # Converting edge IDs instead of weights preserves the link to plasticity.
        indexed = csr_matrix((np.arange(weights._nnz(), dtype=np.int32),
                              weights.col_indices().cpu().numpy(),
                              weights.crow_indices().cpu().numpy()), shape=weights.shape)
        outgoing = indexed.tocsc()
        self.pointers = torch.as_tensor(outgoing.indptr, dtype=torch.int32, device=self.device)
        self.targets = torch.as_tensor(outgoing.indices, dtype=torch.int32, device=self.device)
        self.edge_ids = torch.as_tensor(outgoing.data, dtype=torch.int32, device=self.device)

    def __call__(self, spikes):
        if (spikes.ndim != 2 or spikes.shape[1] != self.size
                or spikes.device != self.device or spikes.dtype != torch.float32
                or not spikes.is_contiguous()):
            raise ValueError('Expected contiguous float32 spikes shaped (batch, neurons) on the weight device')
        if spikes.requires_grad or self.values.requires_grad:
            raise ValueError('Event propagation is only available for inference')
        batch = spikes.shape[0]
        active = torch.empty((batch, self.size), device=self.device, dtype=torch.int32)
        counts = torch.zeros(batch, device=self.device, dtype=torch.int32)
        output = torch.zeros_like(spikes)
        _compact_spikes[(triton.cdiv(self.size, 256), batch)](
            spikes, active, counts, self.size, BLOCK=256)
        _propagate[(256, batch)](
            spikes, active, counts, self.pointers, self.targets, self.edge_ids,
            self.values, output, self.size, WORKERS=256, BLOCK=128)
        return output
