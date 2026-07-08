import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.world import World
from engine.render import Renderer
from engine.rigid_body import RigidBody
from engine.shape import Plane, Box
from engine.math3d import Vector3, Quaternion

if __name__ == "__main__":
    world = World()
    renderer = Renderer(1200, 800)

    ground = RigidBody(Plane(Vector3(0, 1, 0), 0), Vector3(
        0, 0, 0), Quaternion.identity(), 0.0, static=True)
    world.add_body(ground)

    # ピラミッド状に箱を積む
    box_half = 0.5
    rows = 6
    for row in range(rows):
        count = rows - row
        y = box_half + row * box_half * 2
        row_offset = -(count - 1) * box_half
        for i in range(count):
            x = row_offset + i * box_half * 2
            box = RigidBody(Box(box_half, box_half, box_half), Vector3(
                x, y, 0), Quaternion.identity(), 1.0)
            world.add_body(box)

    # 大砲弾(重い箱)を横から高速で撃ち込んでピラミッドを崩す
    cannonball = RigidBody(Box(0.6, 0.6, 0.6), Vector3(-10, box_half, 0),
                            Quaternion.identity(), 8.0)
    cannonball.linear_velocity = Vector3(30, 0, 0)
    world.add_body(cannonball)

    while not renderer.should_close():
        world.step()
        renderer.update_camera()
        renderer.draw_frame(world.bodies)
