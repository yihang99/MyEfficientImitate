import carla
import weakref
import math


class CollisionSensor(object):
    def __init__(self, parent_actor, min_area_of_collision=3, max_area_of_collision=5, max_id_time=5):
        self._parent = parent_actor
        world = self._parent.get_world()
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        col_location = carla.Location(0, 0, 0)
        col_rotation = carla.Rotation(0, 0, 0)
        col_transform = carla.Transform(col_location, col_rotation)
        self.sensor = world.spawn_actor(col_bp, col_transform, attach_to=self._parent,
                                        attachment_type=carla.AttachmentType.Rigid)
        self.sensor.listen(lambda event: self._on_collision(weakref.ref(self), event))

        self.registered_collisions = []
        self.last_id = None
        self.collision_time = None
        self._collision_info = None

        # If closer than this distance, the collision is ignored
        self._min_area_of_collision = min_area_of_collision
        # If further than this distance, the area is forgotten
        self._max_area_of_collision = max_area_of_collision
        # Max amount of time the last collision if is remembered
        self._max_id_time = max_id_time

    def tick(self, timestamp=None):
        ev_loc = self._parent.get_location()
        new_registered_collisions = []
        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:
            distance = ev_loc.distance(collision_location)
            # If far away from a previous collision, forget it
            if distance <= self._max_area_of_collision:
                new_registered_collisions.append(collision_location)

        self.registered_collisions = new_registered_collisions
        if self.last_id and timestamp['relative_simulation_time'] - self.collision_time > self._max_id_time:
            self.last_id = None

        info = self._collision_info
        self._collision_info = None
        if info is not None:
            info['step'] -= timestamp['start_frame']
            info['simulation_time'] -= timestamp['start_simulation_time']
        return info

    def clean(self):
        self.sensor.stop()
        self.sensor.destroy()
        self.sensor = None
        self.registered_collisions = []

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Ignore the current one if it's' the same id as before
        if self.last_id == event.other_actor.id:
            return

        # Ignore if it's too close to a previous collision (avoid micro collisions)
        ev_loc = event.actor.get_transform().location
        for collision_location in self.registered_collisions:
            if ev_loc.distance(collision_location) <= self._min_area_of_collision:
                return

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        # Collision_type
        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
                and 'sidewalk' not in event.other_actor.type_id:
            collision_type = 0  # TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            collision_type = 1  # TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            collision_type = 2  # TrafficEventType.COLLISION_PEDESTRIAN
        else:
            collision_type = -1

        # Write to info, all quantities in in world coordinate
        event_loc = event.transform.location
        event_rot = event.transform.rotation
        oa_loc = event.other_actor.get_transform().location
        oa_rot = event.other_actor.get_transform().rotation
        oa_vel = event.other_actor.get_velocity()
        ev_rot = event.actor.get_transform().rotation
        ev_vel = event.actor.get_velocity()

        self._collision_info = {
            'step': event.frame,
            'simulation_time': event.timestamp,
            'collision_type': collision_type,
            'other_actor_id': event.other_actor.id,
            'other_actor_type_id': event.other_actor.type_id,
            'intensity': intensity,
            'normal_impulse': [impulse.x, impulse.y, impulse.z],
            'event_loc': [event_loc.x, event_loc.y, event_loc.z],
            'event_rot': [event_rot.roll, event_rot.pitch, event_rot.yaw],
            'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z],
            'ev_rot': [ev_rot.roll, ev_rot.pitch, ev_rot.yaw],
            'ev_vel': [ev_vel.x, ev_vel.y, ev_vel.z],
            'oa_loc': [oa_loc.x, oa_loc.y, oa_loc.z],
            'oa_rot': [oa_rot.roll, oa_rot.pitch, oa_rot.yaw],
            'oa_vel': [oa_vel.x, oa_vel.y, oa_vel.z]
        }

        self.collision_time = event.timestamp
        self.registered_collisions.append(ev_loc)

        # Number 0: static objects -> ignore it
        if event.other_actor.id != 0:
            self.last_id = event.other_actor.id
