"""Reuse MuJoCo stages immediately after dm_control's legacy physics step."""

import mujoco


_CALLBACKS = (
    mujoco.get_mjcb_control, mujoco.get_mjcb_passive, mujoco.get_mjcb_sensor,
    mujoco.get_mjcb_contactfilter, mujoco.get_mjcb_act_dyn,
    mujoco.get_mjcb_act_gain, mujoco.get_mjcb_act_bias,
)


def reuse_physics_stages(physics):
    """Refresh acceleration sensors after stepping, reusing current pos/vel data.

    dm_control's legacy step ends in mj_step1, which already updates position
    and velocity stages. Its dirty flag can still be set by control writes,
    causing the next sensor read to repeat those stages with a full forward.
    Complete that refresh here before anything can change the model or state.
    Callbacks, plugins, and the non-legacy step retain the original behavior.
    """
    if getattr(physics, '_stage_reuse_installed', False):
        return
    original_step = physics.step

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if (physics.is_dirty and physics.legacy_step and physics.model.nplugin == 0
                and all(callback() is None for callback in _CALLBACKS)):
            with physics.check_invalid_state():
                mujoco.mj_forwardSkip(physics.model.ptr, physics.data.ptr,
                                     mujoco.mjtStage.mjSTAGE_VEL, 0)
            # Same flag cleared by mjcf.Physics.forward after a full refresh.
            physics._dirty = False
        return result

    physics.step = step
    physics._stage_reuse_installed = True
