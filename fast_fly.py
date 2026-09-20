"""FlyGym observations with compiled-model indices resolved once.

Matches FlyGym 1.0.1's observation and adhesion conventions. Physics still
refreshes dirty state before reading sensors; observations are never reused.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from flygym import Fly


class FastFly(Fly):
    def _observation_indices(self, physics):
        model = physics.model
        if getattr(self, '_observation_model', None) is model:
            return

        def sensor_indices(sensors):
            ids = [model.name2id(sensor.full_identifier, 'sensor') for sensor in sensors]
            return np.concatenate([np.arange(model.sensor_adr[i],
                                            model.sensor_adr[i] + model.sensor_dim[i])
                                   for i in ids]).astype(np.intp)

        self._joint_data_ids = sensor_indices(self._joint_sensors)
        self._body_data_ids = [sensor_indices([sensor]) for sensor in self._body_sensors]
        self._feet_data_ids = sensor_indices(self._end_effector_sensors)
        self._contact_body_ids = np.array([
            model.name2id(name, 'body') for name in self.contact_sensor_placements
        ], dtype=np.intp)
        if self.enable_adhesion:
            self._geom_to_leg = np.full(model.ngeom, -1, dtype=np.intp)
            self._geom_to_leg[self._adhesion_actuator_geom_id] = np.arange(self.n_legs)
        self._observation_model = model

    def get_observation(self, sim):
        # Keep optional native vision/olfaction on the upstream implementation.
        # The apps render eyes separately and use their own olfactory system.
        if self.enable_vision or self.enable_olfaction:
            return super().get_observation(sim)
        physics = sim.physics
        self._observation_indices(physics)
        if physics.is_dirty:
            physics.forward()
        data = physics.data.ptr
        sensors = data.sensordata
        joints = sensors[self._joint_data_ids].reshape(-1, 3).T.copy()
        joints[2] *= 1e-9
        position, velocity, quat, angular_velocity, orientation = (
            sensors[indices] for indices in self._body_data_ids)
        angles = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('ZYX')
        self.last_obs['rot'] = angles
        self.last_obs['pos'] = position

        forces = data.cfrc_ext[self._contact_body_ids, 3:].copy()
        if self.enable_adhesion:
            contacts = data.contact
            geoms = np.column_stack((contacts.geom1, contacts.geom2))
            legs = self._geom_to_leg[geoms]
            rows, sides = np.nonzero((legs >= 0) & (contacts.exclude[:, None] == 0))
            active_legs = legs[rows, sides]
            self._active_adhesion = np.zeros(self.n_legs, dtype=bool)
            self._active_adhesion[active_legs] = True
            # Preserve contact order and average normals per sensor, as FlyGym
            # does when subtracting the adhesion force from measured contact.
            normals = {}
            for row, leg in zip(rows, active_legs):
                sensor = self._adhesion_bodies_with_contact_sensors[leg]
                normals.setdefault(sensor, []).append(contacts.frame[row, :3])
            for sensor, values in normals.items():
                actuator = self._adhesion_bodies_with_contact_sensors == sensor
                if self._last_adhesion[actuator] > 0:
                    forces[sensor] -= self.adhesion_force * np.mean(values, axis=0)

        self.last_obs['contact_forces'] = forces
        self.last_obs['contact_pos'] = data.xpos[self._contact_body_ids].copy().T
        return {
            'joints': joints.astype(np.float32),
            'fly': np.array([position, velocity, angles, angular_velocity], dtype=np.float32),
            'contact_forces': forces.astype(np.float32),
            'end_effectors': sensors[self._feet_data_ids].reshape(self.n_legs, 3).astype(np.float32),
            'fly_orientation': orientation.astype(np.float32),
        }
