"""Single-pass GPU implementation of the existing Hebbian weight update."""

import triton
import triton.language as tl


@triton.jit
def _update(weights, pre_indices, post_indices, activity, signs, lower, upper,
            N: tl.constexpr, ETA: tl.constexpr, ALPHA: tl.constexpr,
            BLOCK: tl.constexpr):
    edge = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = edge < N
    pre = tl.load(pre_indices + edge, valid, other=0)
    post = tl.load(post_indices + edge, valid, other=0)
    pre_rate = tl.load(activity + pre)
    post_rate = tl.load(activity + post)
    weight = tl.load(weights + edge, valid, other=0)
    sign = tl.load(signs + edge, valid, other=0)
    delta = ETA * pre_rate * post_rate * sign - ALPHA * weight
    value = weight + delta
    value = tl.minimum(tl.maximum(value, tl.load(lower + edge, valid, other=0)),
                       tl.load(upper + edge, valid, other=0))
    tl.store(weights + edge, value, valid)


def update_weights(weights, pre, post, activity, signs, lower, upper, eta, alpha):
    if weights.numel():
        _update[(triton.cdiv(weights.numel(), 256),)](
            weights, pre, post, activity, signs, lower, upper,
            weights.numel(), eta, alpha, BLOCK=256, enable_fp_fusion=False)
