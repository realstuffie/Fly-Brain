import numpy as np
import pytest

pytest.importorskip('flygym')
from flygym import Fly
from flygym.examples.locomotion.turning_controller import HybridTurningController
from fast_fly import FastFly


@pytest.mark.parametrize('vision', [False, True])
def test_fly_model_compiles_and_steps(vision):
    contact_sensors = [
        f'{leg}{segment}'
        for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
        for segment in ('Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5')
    ]
    fly = FastFly(enable_adhesion=True, enable_vision=vision,
              contact_sensor_placements=contact_sensors)
    # The app renders eyes separately, with the native MuJoCo renderer.
    fly.enable_vision = False
    sim = HybridTurningController(fly=fly, timestep=1e-4, seed=0)
    try:
        obs, _ = sim.reset(seed=0)
        for _ in range(20):
            obs, _, _, _, _ = sim.step(np.array([1., 1.]))
            assert np.isfinite(obs['fly']).all()
        assert sim.curr_time == pytest.approx(.002)
    finally:
        sim.close()


@pytest.mark.parametrize('adhesion', [False, True])
def test_fast_observations_match_reference_trajectory(adhesion):
    contacts = [f'{leg}{segment}'
                for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
                for segment in ('Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5')]
    simulations = []
    try:
        for fly_type in (Fly, FastFly):
            simulations.append(HybridTurningController(
                fly=fly_type(enable_adhesion=adhesion, contact_sensor_placements=contacts),
                timestep=1e-4, seed=0))
        for sim in simulations:
            sim.reset(seed=0)
        saw_contact = False
        for step in range(600):
            drive = np.array([1., .7 if step > 300 else 1.])
            results = [sim.step(drive) for sim in simulations]
            for key in results[0][0]:
                np.testing.assert_allclose(results[1][0][key], results[0][0][key],
                                           rtol=1e-6, atol=1e-6, err_msg=key)
            saw_contact |= bool(np.any(results[0][0]['contact_forces']))
            if adhesion:
                np.testing.assert_array_equal(simulations[1].fly._active_adhesion,
                                              simulations[0].fly._active_adhesion)
        assert saw_contact
        # Resetting and explicitly dirtying physics must not reuse sensor values.
        for sim in simulations:
            sim.reset(seed=1)
            sim.physics.bind(sim.fly.thorax).pos += np.array([.1, 0., 0.])
        observations = [sim.get_observation() for sim in simulations]
        for key in observations[0]:
            np.testing.assert_allclose(observations[1][key], observations[0][key],
                                       rtol=1e-6, atol=1e-6)
    finally:
        for sim in simulations:
            sim.close()


def test_fast_observations_resolve_each_fly_in_shared_physics():
    from flygym.simulation import Simulation
    flies = [FastFly(name=f'fly{i}', spawn_pos=(0, i * 5, .5)) for i in range(2)]
    sim = Simulation(flies=flies, cameras=[], timestep=1e-4)
    try:
        sim.reset(seed=0)
        actions = {fly.name: {'joints': np.zeros(len(fly.actuated_joints))} for fly in flies}
        for _ in range(20):
            observations, _, _, _, _ = sim.step(actions)
            for fly in flies:
                expected = Fly.get_observation(fly, sim)
                for key in expected:
                    np.testing.assert_array_equal(observations[fly.name][key], expected[key])
    finally:
        sim.close()
