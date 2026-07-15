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

# --- ドミノ ---
DOMINO_HALF_THICKNESS = 0.09
DOMINO_HALF_HEIGHT = 0.5
DOMINO_HALF_WIDTH = 0.3
DOMINO_MASS = 0.4
DOMINO_SPACING = 0.7          # 高さより短くしないと next に届かず倒れ続けない
DOMINO_START = Vector3(0.0, 0.0, -3.0)
STAIR_RISE = 0.2              # 1段の高さ。上げすぎると上り連鎖が途切れる
STAIR_TREAD = 0.6             # 踏み板の奥行き = 階段上のドミノ間隔
STAIR_STEPS = 6
STAIR_HALF_WIDTH = 0.9
BRIDGE_CLEARANCE = 2.4        # プレイヤーの目線(≒2.0)より高くしないと視点が橋にめり込む
# 1段あたりの質量比が約1.5倍(サイズ比1.15倍)を超えると、連鎖の勢いでは
# 押し切れずアーチ状に詰まって止まる(シミュレーションで確認済み)
FINALE_SCALES = (1.15, 1.3, 1.5, 1.7)
# --- ボウリング ---
LANE_X = 20.0
LANE_HALF_WIDTH = 1.45
LANE_FOUL_Z = 15.0            # 投げる側
LANE_PIN_Z = -10.0            # 先頭ピン
PIN_SPACING = 0.55
PIN_MASS = 0.3
PIN_HALF_HEIGHT = 0.45
BOWLING_BALL_RADIUS = 0.42
BOWLING_BALL_MASS = 6.0
PIN_DOWN_THRESHOLD = 0.7      # ピンの上方向とワールド上方向の内積がこれ未満なら倒れたと判定

# --- バスケット ---
HOOP_CENTER = Vector3(-24.0, 3.05, -18.4)
HOOP_RADIUS = 0.78
HOOP_SEGMENTS = 10
BASKET_CATCH_RADIUS = 0.585   # リング中心からこの距離以内をゴールとみなす
BASKETBALL_RADIUS = 0.35
BASKETBALL_MASS = 0.6
BACKBOARD_COLOR = rl.WHITE
HOOP_COLOR = rl.RED


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


def _snapshot(body: RigidBody) -> tuple[RigidBody, Vector3, Quaternion]:
    '''Rキーで復元するために、剛体の初期姿勢を控えておく。'''
    return (body, Vector3(*body.position.as_tuple()),
            Quaternion(body.direct.w, body.direct.x, body.direct.y, body.direct.z))


def _restore(states: list[tuple[RigidBody, Vector3, Quaternion]]) -> None:
    '''全て初期姿勢に戻す。強制sleepは使わず、自然なスリープ判定に任せる。'''
    for body, position, direct in states:
        body.position = Vector3(*position.as_tuple())
        body.direct = Quaternion(direct.w, direct.x, direct.y, direct.z)
        body.linear_velocity = Vector3.zero()
        body.angular_velocity = Vector3.zero()
        body.update_inv_inertia_world()
        body.wake()


def _add_ball(world, states, position: Vector3, radius: float, mass: float, restitution: float) -> RigidBody:
    ball = RigidBody(Sphere(radius), position, Quaternion.identity(), mass)
    ball.restitution = restitution
    world.add_body(ball)
    states.append(_snapshot(ball))
    return ball


def _add_domino_course(world, states) -> list[RigidBody]:
    '''S字→上り階段→高台→下り階段→橋の下→カーブ→巨大化フィナーレのドミノコース。

    「タートル」(現在位置 pos・進行方向 theta・床の高さ floor_y)を進めながら、
    一定間隔ごとにドミノを立てていく。階段や橋の静的ボックスも道なりに置く。
    '''
    dominoes: list[RigidBody] = []
    pos = Vector3(DOMINO_START.x, 0.0, DOMINO_START.z)
    theta = math.pi          # 進行方向。forward = (sinθ, 0, cosθ) なので π は -Z
    floor_y = 0.0
    gap = DOMINO_SPACING     # 直前のドミノからの道のり。先頭にすぐ1枚置く

    def forward() -> Vector3:
        return Vector3(math.sin(theta), 0.0, math.cos(theta))

    def place(scale: float = 1.0) -> None:
        f = forward()
        # 厚み方向(ローカルX軸)を進行方向に向けると、隣へ倒れ込む
        yaw = math.atan2(-f.z, f.x)
        domino = RigidBody(
            Box(DOMINO_HALF_THICKNESS * scale, DOMINO_HALF_HEIGHT * scale,
                DOMINO_HALF_WIDTH * scale),
            Vector3(pos.x, floor_y + DOMINO_HALF_HEIGHT * scale, pos.z),
            Quaternion.from_axis_angle(Vector3(0, 1, 0), yaw),
            DOMINO_MASS * scale ** 3,
        )
        world.add_body(domino)
        states.append(_snapshot(domino))
        dominoes.append(domino)

    def add_static(half_extents: tuple[float, float, float], center: Vector3) -> None:
        '''ローカルX=右、ローカルZ=進行方向 の向きで静的ボックスを置く。'''
        world.add_body(RigidBody(
            Box(*half_extents), center,
            Quaternion.from_axis_angle(Vector3(0, 1, 0), theta), 0.0, static=True))

    def walk(distance: float) -> None:
        nonlocal pos, gap
        remaining = distance
        while remaining > 1e-9:
            ds = min(0.02, remaining)
            pos = pos + forward() * ds
            remaining -= ds
            gap += ds
            if gap >= DOMINO_SPACING:
                place()
                gap = 0.0

    def turn(angle: float, radius: float) -> None:
        '''円弧に沿って曲がりながら進む。angle < 0 で+X側へ曲がる。'''
        nonlocal theta
        arc_length = abs(angle) * radius
        count = max(1, int(arc_length / 0.02))
        for _ in range(count):
            theta += angle / count
            walk(arc_length / count)

    def stairs(rise: float) -> None:
        '''STAIR_STEPS 段の階段。rise 正で上り、負で下り。各踏み板に1枚立てる。'''
        nonlocal pos, floor_y, gap
        if gap > 0.3:
            place()  # 階段の直前に1枚置いて、段差ごしの間隔が空きすぎないようにする
        for _ in range(STAIR_STEPS):
            floor_y += rise
            center = pos + forward() * (STAIR_TREAD / 2)
            if floor_y > 1e-6:  # 地面と同じ高さになった段は箱が不要
                add_static((STAIR_HALF_WIDTH, floor_y / 2, STAIR_TREAD / 2),
                           Vector3(center.x, floor_y / 2, center.z))
            pos = center
            place()
            pos = pos + forward() * (STAIR_TREAD / 2)
        gap = STAIR_TREAD / 2

    def platform(length: float) -> None:
        '''今の床の高さの高台を置き、その上を通常間隔で進む。'''
        center = pos + forward() * (length / 2)
        add_static((STAIR_HALF_WIDTH + 0.1, floor_y / 2, length / 2),
                   Vector3(center.x, floor_y / 2, center.z))
        walk(length)

    def bridge() -> None:
        '''コースをまたぐ橋。デッキの上にはおまけのボールを置いておく。'''
        center = pos + forward() * 0.6
        right = Vector3(math.cos(theta), 0.0, -math.sin(theta))
        for side in (-1.0, 1.0):
            pillar = center + right * (side * 1.5)
            add_static((0.25, BRIDGE_CLEARANCE / 2, 0.4),
                       Vector3(pillar.x, BRIDGE_CLEARANCE / 2, pillar.z))
        deck_center_y = BRIDGE_CLEARANCE + 0.12
        add_static((1.75, 0.12, 0.6), Vector3(center.x, deck_center_y, center.z))
        _add_ball(world, states,
                  Vector3(center.x, deck_center_y + 0.42, center.z),
                  0.3, 1.0, restitution=0.5)
        walk(1.2)  # 橋の下を通過

    def finale() -> None:
        '''1枚ごとに大きくなるドミノで締める。

        間隔は「押す側」の高さに比例させる。次のドミノのサイズ基準で空けると、
        倒れてきた小さいドミノが相手の重心より下しか叩けず、連鎖が止まる。
        '''
        nonlocal pos, gap
        prev_scale = 1.0
        for scale in FINALE_SCALES:
            # gap = 直前のドミノからすでに進んだ道のり。これを差し引かないと間隔が広がりすぎる
            advance = max(0.1, DOMINO_SPACING * prev_scale - gap)
            pos = pos + forward() * advance
            place(scale)
            gap = 0.0
            prev_scale = scale

    walk(3.0)
    turn(math.radians(-60), 3.0)   # S字カーブ(+X側へ)
    turn(math.radians(60), 3.0)    # S字カーブ(戻す)
    walk(1.0)
    stairs(STAIR_RISE)             # 上り階段
    platform(2.0)                  # 高台
    stairs(-STAIR_RISE)            # 下り階段
    walk(1.0)
    bridge()                       # 橋の下をくぐる
    turn(math.radians(-90), 3.5)
    walk(1.5)
    finale()
    return dominoes


def _add_bowling_alley(world, states) -> list[RigidBody]:
    '''ガター壁つきのレーン、10本のピン、重いボールを置く。ピンのリストを返す。'''
    lane_center_z = (LANE_FOUL_Z + LANE_PIN_Z - 3.0) / 2
    lane_half_depth = (LANE_FOUL_Z - (LANE_PIN_Z - 3.0)) / 2
    for side in (-1.0, 1.0):
        gutter = RigidBody(
            Box(0.15, 0.25, lane_half_depth),
            Vector3(LANE_X + side * LANE_HALF_WIDTH, 0.25, lane_center_z),
            Quaternion.identity(), 0.0, static=True,
        )
        world.add_body(gutter)

    backstop = RigidBody(
        Box(LANE_HALF_WIDTH, 1.0, 0.2),
        Vector3(LANE_X, 1.0, LANE_PIN_Z - 3.2),
        Quaternion.identity(), 0.0, static=True,
    )
    world.add_body(backstop)

    pins: list[RigidBody] = []
    for row in range(4):
        for index in range(row + 1):
            pin = RigidBody(
                Box(0.1, PIN_HALF_HEIGHT, 0.1),
                Vector3(LANE_X + (index - row * 0.5) * PIN_SPACING,
                        PIN_HALF_HEIGHT, LANE_PIN_Z - row * 0.5),
                Quaternion.identity(), PIN_MASS,
            )
            world.add_body(pin)
            states.append(_snapshot(pin))
            pins.append(pin)

    for offset in (-0.6, 0.6):
        _add_ball(world, states,
                  Vector3(LANE_X + offset, BOWLING_BALL_RADIUS, LANE_FOUL_Z),
                  BOWLING_BALL_RADIUS, BOWLING_BALL_MASS, restitution=0.1)
    return pins


def _add_basketball_hoop(world, states) -> list[RigidBody]:
    '''ポール・バックボード・リング(静的な箱の輪)を組み、ボールを置く。ボールのリストを返す。'''
    pole = RigidBody(
        Box(0.15, 1.75, 0.15),
        Vector3(HOOP_CENTER.x, 1.75, HOOP_CENTER.z - 1.2),
        Quaternion.identity(), 0.0, static=True,
    )
    world.add_body(pole)

    backboard = RigidBody(
        Box(1.3, 0.8, 0.08),
        Vector3(HOOP_CENTER.x, HOOP_CENTER.y + 0.45, HOOP_CENTER.z - 0.8),
        Quaternion.identity(), 0.0, static=True,
    )
    backboard.color = BACKBOARD_COLOR
    world.add_body(backboard)

    for i in range(HOOP_SEGMENTS):
        angle = 2.0 * math.pi * i / HOOP_SEGMENTS
        segment = RigidBody(
            Box(0.06, 0.04, math.pi * HOOP_RADIUS / HOOP_SEGMENTS),
            Vector3(HOOP_CENTER.x + HOOP_RADIUS * math.cos(angle),
                    HOOP_CENTER.y,
                    HOOP_CENTER.z + HOOP_RADIUS * math.sin(angle)),
            # ローカルZ(長辺)を円の接線方向に向ける
            Quaternion.from_axis_angle(Vector3(0, 1, 0), -angle),
            0.0, static=True,
        )
        segment.color = HOOP_COLOR
        world.add_body(segment)

    balls = []
    for offset in (-1.5, 1.5):
        balls.append(_add_ball(
            world, states,
            Vector3(HOOP_CENTER.x + offset, BASKETBALL_RADIUS, HOOP_CENTER.z + 3.5),
            BASKETBALL_RADIUS, BASKETBALL_MASS, restitution=0.7))
    return balls


def _count_standing(bodies: list[RigidBody]) -> int:
    '''上方向ベクトルの傾きで、まだ立っている剛体を数える(ピン・ドミノ共通)。'''
    return sum(1 for body in bodies
               if body.direct.rotate_vector(Vector3(0, 1, 0)).y >= PIN_DOWN_THRESHOLD)


def _passed_through_hoop(ball: RigidBody, previous_height: float) -> bool:
    '''リング面を上から下へ通過した瞬間だけ True。'''
    if not (previous_height > HOOP_CENTER.y >= ball.position.y):
        return False
    dx = ball.position.x - HOOP_CENTER.x
    dz = ball.position.z - HOOP_CENTER.z
    return math.hypot(dx, dz) < BASKET_CATCH_RADIUS


def _draw_scene(camera, world: World, player: RigidBody, dominoes: list[RigidBody],
                pins: list[RigidBody], basket_score: int) -> None:
    rl.begin_drawing()
    rl.clear_background(rl.SKYBLUE)
    rl.begin_mode_3d(camera)
    draw_bodies([body for body in world.bodies if body is not player])
    rl.end_mode_3d()
    rl.draw_text(
        "WASD: move   Mouse: look   Space: jump   E: hold/throw ball   R: reset   Esc: quit",
        10, 10, 20, rl.BLACK)
    rl.draw_text(
        f"Dominoes: {len(dominoes) - _count_standing(dominoes)}/{len(dominoes)}   "
        f"Pins down: {10 - _count_standing(pins)}/10   Basket: {basket_score}",
        10, 40, 20, rl.BLACK)
    rl.draw_fps(10, 70)
    rl.end_drawing()


if __name__ == "__main__":
    world = World()
    rl.init_window(1280, 720, "3D Physics Engine - First Person Walk")
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

    # --- 遊べる要素 ---
    initial_states: list[tuple[RigidBody, Vector3, Quaternion]] = []
    dominoes = _add_domino_course(world, initial_states)
    pins = _add_bowling_alley(world, initial_states)
    basketballs = _add_basketball_hoop(world, initial_states)

    # プレイヤーの足元に置く、ドミノを倒すための投げ玉
    for offset in (-1.0, 0.0, 1.0):
        _add_ball(world, initial_states, Vector3(3.0 + offset, 0.3, 4.0),
                  0.3, 1.0, restitution=0.5)

    player = RigidBody(Sphere(PLAYER_RADIUS), Vector3(
        0.0, 2.0, 5.0), Quaternion.identity(), mass=PLAYER_MASS)
    player.restitution = 0.0
    player.can_sleep = False  # 速度を直接書き込んで動かすので、眠ると入力が効かなくなる
    world.add_body(player)

    yaw_deg = 180.0
    pitch_deg = 0.0
    held_body: RigidBody | None = None
    basket_score = 0
    previous_ball_heights = [ball.position.y for ball in basketballs]

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

        if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            _restore(initial_states)
            held_body = None
            basket_score = 0
            previous_ball_heights = [ball.position.y for ball in basketballs]

        if rl.is_key_pressed(rl.KeyboardKey.KEY_E):
            if held_body is None:
                held_body = _find_pickup_target(
                    world, player, eye, look_dir, PICKUP_RANGE)
            else:
                held_body.linear_velocity = look_dir * THROW_SPEED
                held_body = None

        if held_body is not None:
            held_body.wake()  # 位置を直接動かすので、スリープさせない
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

        for index, ball in enumerate(basketballs):
            if _passed_through_hoop(ball, previous_ball_heights[index]):
                basket_score += 1
            previous_ball_heights[index] = ball.position.y

        eye = Vector3(player.position.x, player.position.y +
                      PLAYER_EYE_HEIGHT, player.position.z)
        look_target = eye + _look_direction(yaw_deg, pitch_deg)
        camera.position = rl.Vector3(eye.x, eye.y, eye.z)
        camera.target = rl.Vector3(look_target.x, look_target.y, look_target.z)

        _draw_scene(camera, world, player, dominoes, pins, basket_score)

        # print(f"FPS: {renderer.get_fps()}")

    rl.close_window()
