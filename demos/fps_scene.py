import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.world import World
from engine.render import Renderer, draw_bodies
from engine.rigid_body import RigidBody
from engine.shape import Plane, Box, Sphere
from engine.math3d import *
import pyray as rl


PLAYER_RADIUS = 0.4
PLAYER_MASS = 5.0
PLAYER_EYE_HEIGHT = 1.6
MOVE_SPEED = 10.0
JUMP_SPEED = 4.0
MOUSE_SENSITIVITY = 0.15
PITCH_LIMIT_DEG = 89.0
FIXED_DT = 1.0 / 120.0
GROUND_NORMAL_THRESHOLD = 0.5
PICKUP_RANGE = 6.0
PICKUP_AIM_TOLERANCE = 0.6
HOLD_DISTANCE = 2.2
THROW_SPEED = 18.0


def _is_grounded(world: World, player: RigidBody) -> bool:
    for manifold in world.last_contacts:
        if manifold.body_a is player and manifold.normal.y > GROUND_NORMAL_THRESHOLD:
            return True
        if manifold.body_b is player and -manifold.normal.y > GROUND_NORMAL_THRESHOLD:
            return True
    return False


def _look_direction(yaw_deg: float, pitch_deg: float) -> Vector3:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return Vector3(
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    )


def _find_pickup_target(world: World, player: RigidBody, eye: Vector3, look_dir: Vector3, max_distance: float):
    '''視線方向の先にあるボール(Sphere)のうち、一番手前のものを返す。'''
    best_body = None
    best_distance = max_distance
    for body in world.bodies:
        if body is player or body.is_static or not isinstance(body.shape, Sphere):
            continue
        to_body = body.position - eye
        along_view = to_body.dot(look_dir)
        if along_view <= 0.0:
            continue
        closest_point = eye + look_dir * along_view
        aim_offset = (body.position - closest_point).length()
        if aim_offset <= body.shape.radius + PICKUP_AIM_TOLERANCE and along_view < best_distance:
            best_body = body
            best_distance = along_view
    return best_body


if __name__ == "__main__":
    world = World()
    rl.init_window(1280, 720, "3D Physics Engine - FPS Walk")
    rl.set_target_fps(60)
    rl.disable_cursor()

    ground = RigidBody(Plane(Vector3(0, 1, 0), 0), Vector3(
        0, 0, 0), Quaternion.identity(), 0.0, static=True)
    world.add_body(ground)

    # --- 外周の壁 ---
    ARENA_HALF_SIZE = 35.0
    WALL_HALF_HEIGHT = 3.0
    WALL_HALF_THICKNESS = 0.5

    wall_specs = [
        ((ARENA_HALF_SIZE, WALL_HALF_HEIGHT, WALL_HALF_THICKNESS),
         Vector3(0, WALL_HALF_HEIGHT, ARENA_HALF_SIZE)),
        ((ARENA_HALF_SIZE, WALL_HALF_HEIGHT, WALL_HALF_THICKNESS),
         Vector3(0, WALL_HALF_HEIGHT, -ARENA_HALF_SIZE)),
        ((WALL_HALF_THICKNESS, WALL_HALF_HEIGHT, ARENA_HALF_SIZE),
         Vector3(ARENA_HALF_SIZE, WALL_HALF_HEIGHT, 0)),
        ((WALL_HALF_THICKNESS, WALL_HALF_HEIGHT, ARENA_HALF_SIZE),
         Vector3(-ARENA_HALF_SIZE, WALL_HALF_HEIGHT, 0)),
    ]
    for half_extents, position in wall_specs:
        wall = RigidBody(Box(*half_extents), position,
                          Quaternion.identity(), 0.0, static=True)
        world.add_body(wall)

    # --- 坂道とその先のプラットフォーム ---
    RAMP_HALF_LENGTH = 5.0
    RAMP_HALF_WIDTH = 3.0
    RAMP_HALF_THICKNESS = 0.3
    RAMP_RISE = 4.0
    RAMP_X = -12.0
    RAMP_Z_NEAR = -6.0

    ramp_angle = math.asin(RAMP_RISE / (2 * RAMP_HALF_LENGTH))
    ramp_run = 2 * RAMP_HALF_LENGTH * math.cos(ramp_angle)
    ramp_center_z = RAMP_Z_NEAR + ramp_run / 2
    ramp_orientation = Quaternion.from_axis_angle(Vector3(1, 0, 0), -ramp_angle)

    ramp = RigidBody(
        Box(RAMP_HALF_WIDTH, RAMP_HALF_THICKNESS, RAMP_HALF_LENGTH),
        Vector3(RAMP_X, RAMP_RISE / 2, ramp_center_z),
        ramp_orientation, 0.0, static=True,
    )
    world.add_body(ramp)

    PLATFORM_HALF_WIDTH = 4.0
    PLATFORM_HALF_THICKNESS = 0.3
    PLATFORM_HALF_DEPTH = 4.0
    platform_z_start = RAMP_Z_NEAR + ramp_run
    platform = RigidBody(
        Box(PLATFORM_HALF_WIDTH, PLATFORM_HALF_THICKNESS, PLATFORM_HALF_DEPTH),
        Vector3(RAMP_X, RAMP_RISE - PLATFORM_HALF_THICKNESS,
                platform_z_start + PLATFORM_HALF_DEPTH),
        Quaternion.identity(), 0.0, static=True,
    )
    world.add_body(platform)

    for i in range(30):
        box = RigidBody(Box(0.1, 1, 0.5), Vector3(
            1*i, 1, 0), Quaternion.identity(), 1.0)
        world.add_body(box)

    angle = Quaternion.from_axis_angle(Vector3(0, 0, 1), -10)

    first_box = RigidBody(Box(0.1, 1, 0.5), Vector3(
        1*(-1), 0.9, 0), angle, 1.0)
    world.add_body(first_box)

    sphere = RigidBody(Sphere(0.5), Vector3(
        3.1, 3.0, 0.0), Quaternion.identity())
    world.add_body(sphere)

    player = RigidBody(Sphere(PLAYER_RADIUS), Vector3(
        0.0, 2.0, 5.0), Quaternion.identity(), mass=PLAYER_MASS)
    player.restitution = 0.0
    world.add_body(player)

    yaw_deg = 180.0
    pitch_deg = 0.0
    held_body: RigidBody | None = None

    camera = rl.Camera3D(
        rl.Vector3(player.position.x, player.position.y +
                   PLAYER_EYE_HEIGHT, player.position.z),
        rl.Vector3(0.0, 0.0, 0.0),
        rl.Vector3(0.0, 1.0, 0.0),
        60.0,
        rl.CameraProjection.CAMERA_PERSPECTIVE,
    )

    while not rl.window_should_close():

        delta = rl.get_mouse_delta()
        yaw_deg -= delta.x * MOUSE_SENSITIVITY
        pitch_deg = max(-PITCH_LIMIT_DEG, min(PITCH_LIMIT_DEG,
                                              pitch_deg - delta.y * MOUSE_SENSITIVITY))

        yaw = math.radians(yaw_deg)
        forward = Vector3(math.sin(yaw), 0.0, math.cos(yaw))
        right = Vector3(math.cos(yaw), 0.0, -math.sin(yaw))
        look_dir = _look_direction(yaw_deg, pitch_deg)

        eye = Vector3(player.position.x, player.position.y +
                      PLAYER_EYE_HEIGHT, player.position.z)

        if rl.is_key_pressed(rl.KeyboardKey.KEY_E):
            if held_body is None:
                held_body = _find_pickup_target(
                    world, player, eye, look_dir, PICKUP_RANGE)
            else:
                held_body.linear_velocity = look_dir * THROW_SPEED
                held_body = None

        if held_body is not None:
            held_body.position = eye + look_dir * HOLD_DISTANCE
            held_body.linear_velocity = Vector3.zero()
            held_body.angular_velocity = Vector3.zero()

        move = Vector3.zero()
        if rl.is_key_down(rl.KeyboardKey.KEY_W):
            move = move + forward
        if rl.is_key_down(rl.KeyboardKey.KEY_S):
            move = move - forward
        if rl.is_key_down(rl.KeyboardKey.KEY_D):
            move = move - right
        if rl.is_key_down(rl.KeyboardKey.KEY_A):
            move = move + right
        if move.length() > 1e-6:
            move = move.normalized()

        player.linear_velocity = Vector3(
            move.x * MOVE_SPEED, player.linear_velocity.y, move.z * MOVE_SPEED
        )

        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) and _is_grounded(world, player):
            player.linear_velocity = Vector3(
                player.linear_velocity.x, JUMP_SPEED, player.linear_velocity.z
            )

        world.step(FIXED_DT)
        player.angular_velocity = Vector3.zero()

        eye = Vector3(player.position.x, player.position.y +
                      PLAYER_EYE_HEIGHT, player.position.z)
        look_target = eye + _look_direction(yaw_deg, pitch_deg)
        camera.position = rl.Vector3(eye.x, eye.y, eye.z)
        camera.target = rl.Vector3(look_target.x, look_target.y, look_target.z)

        rl.begin_drawing()
        rl.clear_background(rl.SKYBLUE)
        rl.begin_mode_3d(camera)
        draw_bodies([body for body in world.bodies if body is not player])
        rl.end_mode_3d()
        rl.draw_text(
            "WASD: move   Mouse: look   Space: jump   E: hold/throw ball   Esc: quit",
            10, 10, 20, rl.BLACK)
        rl.draw_fps(10, 40)
        rl.end_drawing()

        # print(f"FPS: {renderer.get_fps()}")
