import pytest
import torch

import brain_body_bridge as bridge_module
from brain_body_bridge import BrainEngine


@pytest.fixture
def small_brain(monkeypatch, tmp_path):
    ids = list(bridge_module.DN_NEURONS.values())[:2]
    monkeypatch.setattr(bridge_module, 'get_hash_tables', lambda path: (
        {flyid: i for i, flyid in enumerate(ids)}, dict(enumerate(ids))))
    monkeypatch.setattr(bridge_module, 'get_weights', lambda *args: (
        torch.tensor([[0., -2.], [3., 0.]]).to_sparse_csr()))

    def create(path=None):
        return BrainEngine(device='cpu',
                           plastic_path=path or tmp_path / 'weights.pt')
    return create


def test_batched_block_matches_per_step_readout(small_brain):
    batched, reference = small_brain(), small_brain()
    for brain in (batched, reference):
        brain.rates.fill_(2000.)
        brain.register_population('pop', [0, 1])
    body_steps, steps_per_body = 203, 5
    batched_readouts = []
    torch.manual_seed(42)
    for _ in range(body_steps):
        batched.queue_steps(steps_per_body)
        while batched.run_block():
            pass
        batched_readouts.extend(batched.drain_spike_readouts())
    reference_readouts = []
    torch.manual_seed(42)
    for _ in range(body_steps):
        for _ in range(steps_per_body):
            reference.step()
            reference_readouts.append(reference.get_spike_readout())
    assert batched.pending_spike_readouts() == 0
    assert batched._step_debt == 0
    assert len(batched_readouts) == len(reference_readouts)
    for actual, expected in zip(batched_readouts, reference_readouts):
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
    for a, b in zip(batched.state, reference.state):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(batched._syn_vals, reference._syn_vals,
                               rtol=0, atol=0)
    assert any(row[1]['pop'] > 0 for row in batched_readouts)


def test_input_change_finishes_pending_steps(small_brain):
    brain = small_brain()
    brain.rates.fill_(2000.)
    brain.queue_steps(3)
    brain.set_stimulus(None)
    assert brain._step_debt == 0
    assert brain.pending_spike_readouts() == 3
    brain.finish_pending_steps()
    assert brain.pending_spike_readouts() == 3
    brain.drain_spike_readouts()
    assert brain.pending_spike_readouts() == 0


def test_register_population_rejects_queued_readouts(small_brain):
    brain = small_brain()
    brain.queue_steps(1)
    assert brain.run_block()
    with pytest.raises(RuntimeError, match='queued spike readouts'):
        brain.register_population('pop', [0, 1])
