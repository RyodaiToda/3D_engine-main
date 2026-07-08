from .math3d import *
import pyray as rl
from .rigid_body import RigidBody
from .shape import Box, Sphere, Plane

_MOUSE_SENSITIVITY = 0.2  # 度/ピクセル(回転)
_PAN_SENSITIVITY = 0.0015  # distanceに対する係数(平行移動)
_ZOOM_SPEED = 1.0
_MIN_DISTANCE = 2.0
_MAX_DISTANCE = 60.0
_PITCH_LIMIT_DEG = 89.0

_checkered_ball_model = None

def _quaternion_to_axis_angle_degrees(orientation: Quaternion) -> tuple[Vector3, float]:
    """四元数を rl_rotatef が要求する (回転軸, 角度[度]) に変換する。"""
    q = orientation.normalized()
    w = max(-1.0, min(1.0, q.w))
    angle_rad = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-6:
        axis = Vector3(1.0, 0.0, 0.0)
    else:
        axis = Vector3(q.x / s, q.y / s, q.z / s)
    return axis, math.degrees(angle_rad)


def _draw_box_body(body: RigidBody, color) -> None:
    box: Box = body.shape
    axis, angle_deg = _quaternion_to_axis_angle_degrees(body.direct)

    rl.rl_push_matrix()
    rl.rl_translatef(body.position.x, body.position.y, body.position.z)
    rl.rl_rotatef(angle_deg, axis.x, axis.y, axis.z)
    width, height, length = box.half_width * \
        2, box.half_height * 2, box.half_depth * 2
    origin = rl.Vector3(0.0, 0.0, 0.0)
    rl.draw_cube(origin, width, height, length, color)
    rl.draw_cube_wires(origin, width, height, length, rl.BLACK)
    rl.rl_pop_matrix()


def _get_checkered_ball_model():
    """半径1の球に市松模様のテクスチャを貼ったModelを、初回だけ生成して使い回す。

    単色の球は自転しても見た目が変わらないため、回転を視認できるように
    テクスチャの模様が一緒に回る Model 描画(draw_model_ex)を使う。
    """
    global _checkered_ball_model
    if _checkered_ball_model is None:
        image = rl.gen_image_checked(64, 64, 8, 8, rl.WHITE, rl.LIGHTGRAY)
        texture = rl.load_texture_from_image(image)
        rl.unload_image(image)
        mesh = rl.gen_mesh_sphere(1.0, 16, 16)
        model = rl.load_model_from_mesh(mesh)
        rl.set_material_texture(
            model.materials[0], rl.MaterialMapIndex.MATERIAL_MAP_ALBEDO, texture)
        _checkered_ball_model = model
    return _checkered_ball_model


def _draw_sphere_body(body: RigidBody, color) -> None:
    sphere: Sphere = body.shape
    model = _get_checkered_ball_model()
    axis, angle_deg = _quaternion_to_axis_angle_degrees(body.direct)
    position = rl.Vector3(body.position.x, body.position.y, body.position.z)
    scale = rl.Vector3(sphere.radius, sphere.radius, sphere.radius)
    rl.draw_model_ex(model, position, rl.Vector3(
        axis.x, axis.y, axis.z), angle_deg, scale, color)


def draw_bodies(bodies: list[RigidBody], ground_half_size: float = 25.0) -> None:
    for body in bodies:
        if isinstance(body.shape, Box):
            color = rl.GRAY if body.is_static else rl.MAROON
            _draw_box_body(body, color)
        elif isinstance(body.shape, Sphere):
            _draw_sphere_body(body, rl.ORANGE)
        elif isinstance(body.shape, Plane):
            rl.draw_grid(int(ground_half_size * 2), 1.0)


class Renderer:
    def __init__(self, width, height):

        rl.init_window(width, height,"3D Physics Simulation")
        rl.set_target_fps(60)
        self.width = width
        self.height = height
        self.canvas = [[(255, 255, 255) for _ in range(width)]
                       for _ in range(height)]

        self.target = Vector3(0, 0, 0)
        self.distance = 10.0
        self.yaw_deg = -50.0
        self.pitch_deg = 25.0

        self.camera = rl.Camera3D(
            rl.Vector3(0, 0, 0),
            rl.Vector3(self.target.x, self.target.y, self.target.z),
            rl.Vector3(0, 1, 0),
            45.0,
            rl.CameraProjection.CAMERA_PERSPECTIVE,
        )

        self._update_camera_position()

    def _forward_offset(self) -> Vector3:
        """target→カメラ の向き(オフセット)の単位ベクトルを返す。"""
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        return Vector3(
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.cos(yaw),
        )

    def update_camera(self) -> None:
        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
            delta = rl.get_mouse_delta()
            self.yaw_deg -= delta.x * _MOUSE_SENSITIVITY
            self.pitch_deg = max(-_PITCH_LIMIT_DEG, min(_PITCH_LIMIT_DEG,
                                 self.pitch_deg - delta.y * _MOUSE_SENSITIVITY))

        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
            delta = rl.get_mouse_delta()
            right, up = self._right_up_vectors()
            pan_scale = self.distance * _PAN_SENSITIVITY
            self.target = self.target - right * \
                (delta.x * pan_scale) + up * (delta.y * pan_scale)

        wheel = rl.get_mouse_wheel_move()
        if wheel != 0:
            self.distance = max(_MIN_DISTANCE, min(
                _MAX_DISTANCE, self.distance - wheel * _ZOOM_SPEED))

        self._update_camera_position()

    def _right_up_vectors(self) -> tuple[Vector3, Vector3]:
        """パン操作用に、画面の右方向・上方向に対応するワールド空間ベクトルを求める。"""
        forward = self._forward_offset() * -1.0  # カメラ→target の向き
        world_up = Vector3(0.0, 1.0, 0.0)
        right = forward.cross(world_up).normalized()
        up = right.cross(forward)
        return right, up

    def _update_camera_position(self) -> None:
        offset = self._forward_offset() * self.distance
        self.camera.position = rl.Vector3(
            self.target.x + offset.x, self.target.y + offset.y, self.target.z + offset.z
        )
        self.camera.target = rl.Vector3(
            self.target.x, self.target.y, self.target.z)

    def draw_frame(self, bodies: list[RigidBody]) -> None:
        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        rl.begin_mode_3d(self.camera)
        draw_bodies(bodies)

        rl.end_mode_3d()
        rl.draw_fps(10, 10)
        rl.end_drawing()

    def should_close(self) -> bool:
        return rl.window_should_close()

    def get_fps(self) -> int:
        return rl.get_fps()
