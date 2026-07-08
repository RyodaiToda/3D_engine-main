import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.world import World
from engine.render import Renderer
from engine.rigid_body import RigidBody
from engine.shape import Plane, Box, Sphere
from engine.math3d import *

if __name__ == "__main__":
    world = World()
    renderer = Renderer(1200, 800)

    ground = RigidBody(Plane(Vector3(0, 1, 0), 0), Vector3(
        0, 0, 0), Quaternion.identity(), 0.0, static=True)
    world.add_body(ground)

    # box = RigidBody(Box(1, 1, 1), Vector3(0, 3, 0), Quaternion.identity(), 1.0)
    # world.add_body(box)

    # box = RigidBody(Box(1, 1, 1), Vector3(0, 10, 0), Quaternion.identity(), 1.0)
    # world.add_body(box)

    for i in range(30):
        box = RigidBody(Box(0.1, 1, 0.5), Vector3(
            1*i, 1, 0), Quaternion.identity(), 1.0)
        world.add_body(box)

    angle = Quaternion.from_axis_angle(Vector3(0, 0, 1), -10)

    first_box = RigidBody(Box(0.1, 1, 0.5), Vector3(
        1*(-1), 0.9, 0), angle, 1.0)
    world.add_body(first_box)

    sphere = RigidBody(Sphere(0.5), Vector3(3.1, 3.0, 0.0),Quaternion.identity())
    world.add_body(sphere)

    while not renderer.should_close():

        # print(f"FPS: {renderer.get_fps()}")
        world.step()
        renderer.update_camera()
        renderer.draw_frame(world.bodies)
