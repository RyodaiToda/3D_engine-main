
import math
from .math3d import *

# --- 物理定数 ---
AIR_RESISTANCE = 0.995
ROT_RESISTANCE = 0.995
RESTITUTION = 0.45
FRICTION = 0.50


class RigidBody:
    def __init__(self, shape, position: Vector3, direct: Quaternion,  mass: float = 1.0, static=False):
        self.shape = shape
        self.position = position
        # self.size = Vector3(w, h, d)

        # self.color = color
        self.direct = direct
        self.linear_velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()

        self.force_accum = Vector3.zero()
        self.torque_accum = Vector3.zero()

        self.restitution = RESTITUTION
        self.friction = FRICTION

        self.vel = Vector3(0, 0, 0)
        self.angular_momentum = Vector3(0, 0, 0)

        self.is_static = static

        # スリープ: 一定時間ほぼ静止した剛体は計算から外す(接触で起こされる)
        self.is_sleeping = False
        self.can_sleep = True
        self.sleep_timer = 0.0
        self._aabb_cache = None  # static/sleeping中のブロードフェーズAABBキャッシュ

        if self.is_static:
            self.mass = float('inf')
            self.inv_mass = 0.0
            self.inertia_body = Matrix3.identity()
            self.inv_inertia_body = Matrix3.diagonal(0.0, 0.0, 0.0)
        else:
            self.mass = mass
            self.inv_mass = 1.0 / mass
            self.inertia_body = self.shape.inertia_tensor(mass)
            # print(self.inertia_body)
            # exit()
            self.inv_inertia_body = _inverse_diagonal(self.inertia_body)

        self.inv_inertia_world = Matrix3.diagonal(0.0, 0.0, 0.0)
        self.update_inv_inertia_world()

    def update_inv_inertia_world(self):
        if self.is_static:
            self.inv_inertia_world = Matrix3.diagonal(0.0, 0.0, 0.0)
        else:
            rotation_matrix = self.direct.to_matrix3()
            self.inv_inertia_world = rotation_matrix * \
                self.inv_inertia_body * rotation_matrix.transpose()

    def wake(self):
        '''スリープを解除する。位置を直接書き換えた後にも呼ぶこと。'''
        if not self.is_static:
            self.is_sleeping = False
            self.sleep_timer = 0.0
            # スリープ中は inv_mass=0(不動)にしているので、元に戻す
            self.inv_mass = 1.0 / self.mass
            self.update_inv_inertia_world()
        self._aabb_cache = None

    def sleep(self):
        '''スリープさせる。速度を捨て、ソルバーからは静的物体と同じに見える。'''
        self.is_sleeping = True
        self.sleep_timer = 0.0
        self.linear_velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()
        self.inv_mass = 0.0
        self.inv_inertia_world = Matrix3.diagonal(0.0, 0.0, 0.0)
        self._aabb_cache = None  # 最終位置でAABBを取り直す

    def apply_force(self, force):
        self.force_accum += force

    def apply_torque(self, torque):
        self.angular_momentum += torque

    def clear_accumulators(self):
        self.force_accum = Vector3.zero()
        self.torque_accum = Vector3.zero()

    def integrate_velocity(self, dt):
        if self.is_static:
            return

        self.linear_velocity += (self.force_accum *
                                 self.inv_mass) * dt  # 並進速度の更新
        self.angular_velocity += self.torque_accum * dt  # 角速度の更新

    def integrate_position(self, dt):
        if self.is_static:
            return

        self.position += self.linear_velocity*dt
        self.direct = self.direct.integrate(self.angular_velocity,dt)
        self.update_inv_inertia_world()


def _inverse_diagonal(m: Matrix3) -> Matrix3:
    # print(m)
    """対角慣性テンソルの逆行列(対角成分の逆数)を返す。"""
    ixx, iyy, izz = m.m[0][0], m.m[1][1], m.m[2][2]
    return Matrix3.diagonal(
        0.0 if ixx == 0.0 else 1.0 / ixx,
        0.0 if iyy == 0.0 else 1.0 / iyy,
        0.0 if izz == 0.0 else 1.0 / izz,
    )
