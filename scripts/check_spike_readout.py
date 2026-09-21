"""Compare old and combined readouts on the same full-connectome GPU trajectory."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_body_bridge import BrainEngine, DNRateDecoder, BrainBodyBridge
from visual_system import VisualSystem
from somatosensory import SomatosensorySystem


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires the GPU fast path')
    torch.manual_seed(42)
    brain = BrainEngine(propagation='event')
    old, new = DNRateDecoder(), DNRateDecoder()
    drives = [BrainBodyBridge(decoder) for decoder in (old, new)]
    # Also cover the applications' no-population branch before registration.
    assert brain.get_spike_readout() == (brain.get_dn_spikes(), {})
    visual = VisualSystem(brain.flyid2i, brain.i2flyid)
    somato = SomatosensorySystem(brain.flyid2i)
    populations = {**visual.get_lplc2_indices(brain.flyid2i),
                   **visual.get_lc4_indices(brain.flyid2i)}
    populations.update(JO_touch_L=somato.touch_idx_left,
                       JO_touch_R=somato.touch_idx_right,
                       JO_sound_L=somato.sound_idx_left,
                       JO_sound_R=somato.sound_idx_right)
    for name, indices in populations.items():
        brain.register_population(name, indices)
        old.register_population(name)
        new.register_population(name)
    initial_weights = brain._syn_vals.clone()
    total_spikes = 0
    dn_spikes = 0
    for step in range(1200):
        if step % 200 == 0:
            brain.set_stimulus(['p9', 'sugar', 'lc4', 'jo', 'bitter', 'or56a'][step // 200])
        brain.step()
        tensors = (*brain.state, brain.rates, brain._syn_vals, brain._spike_acc)
        versions = [tensor._version for tensor in tensors]
        rng = torch.cuda.get_rng_state()
        expected = brain.get_dn_spikes(), brain.get_population_spikes()
        actual = brain.get_spike_readout()
        assert actual == expected, (step, actual, expected)
        assert [tensor._version for tensor in tensors] == versions
        assert torch.equal(torch.cuda.get_rng_state(), rng)
        assert brain.pending_spike_readouts() == 0 and brain._step_debt == 0
        old.update(*expected)
        new.update(*actual)
        assert old.rates == new.rates and old.pop_rates == new.pop_rates
        np.testing.assert_array_equal(drives[0].compute_drive(dt=.01),
                                      drives[1].compute_drive(dt=.01))
        assert drives[0].mode == drives[1].mode
        total_spikes += int(brain.state[2].sum().item())
        dn_spikes += int(sum(actual[0].values()))
    assert total_spikes > 0 and dn_spikes > 0
    assert not torch.equal(initial_weights, brain._syn_vals)
    print(json.dumps(dict(result='PASS', brain_steps=1200, neural_ms=120,
                          neurons=brain.num_neurons, synapses=brain._syn_vals.numel(),
                          populations=len(populations), total_spikes=total_spikes,
                          dn_spikes=dn_spikes, readout_mismatches=0,
                          decoder_mismatches=0, drive_mismatches=0,
                          readout_mutations=0, saved_weights=False), indent=2))


if __name__ == '__main__':
    main()
