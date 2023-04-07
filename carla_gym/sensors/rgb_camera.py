import carla
import weakref
from queue import Queue, Empty
import numpy as np
import copy


class RgbCameraSensor(object):
    def __init__(self, parent_actor, fov=90, image_width=300, image_height=200, x=1, y=0, z=2.5):
        self._parent = parent_actor
        self._world = self._parent.get_world()
        cam_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(image_width))
        cam_bp.set_attribute("image_size_y", str(image_height))
        cam_bp.set_attribute("fov", str(fov))
        cam_location = carla.Location(x, y, z)
        cam_rotation = carla.Rotation(0, 180, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        self.sensor = self._world.spawn_actor(cam_bp, cam_transform, attach_to=self._parent,
                                              attachment_type=carla.AttachmentType.Rigid)
        self.sensor.listen(lambda image: self._parse_image(weakref.ref(self), image))

        self._image_queue = Queue()
        self._queue_timeout = 10.0

    def tick(self):
        snap_shot = self._world.get_snapshot()
        assert self._image_queue.qsize() <= 1

        try:
            frame, data = self._image_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('RGB sensor took too long!')

        obs = {'frame': frame,
               'data': data}

        return obs

    @staticmethod
    def _parse_image(weak_self, carla_image):
        self = weak_self()

        np_img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))

        np_img = np.reshape(np_img, (carla_image.height, carla_image.width, 4))
        np_img = np_img[:, :, :3]
        np_img = np_img[:, :, ::-1]

        self._image_queue.put((carla_image.frame, np_img))

    def clean(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None
        self._world = None
        self._image_queue = None
