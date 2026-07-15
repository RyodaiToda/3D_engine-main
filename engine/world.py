from itertools import combinations

from .collision import ContactManifold, find_collision_pairs
from .rigid_body import RigidBody
from .math3d import Vector3, Matrix3
from .collision import collide
from .solver import resolve_velocities

# --- スリープ判定 ---
_SLEEP_LINEAR_THRESHOLD = 0.12   # これ未満の並進速度が続いたら静止とみなす [m/s]
_SLEEP_ANGULAR_THRESHOLD = 0.25  # [rad/s]
_SLEEP_TIME_REQUIRED = 0.5       # 静止がこの秒数続いたらスリープ

# 起こす閾値はスリープ閾値より高くする。静止しかけの物体が触れただけで
# 隣を起こすと、積み重なった山が永遠に眠れなくなる
_WAKE_LINEAR_THRESHOLD = 0.25
_WAKE_ANGULAR_THRESHOLD = 0.5


def _is_disturbing(body: RigidBody) -> bool:
    '''スリープ中の隣の剛体を起こすほど動いているか。'''
    return (body.linear_velocity.length_squared() > _WAKE_LINEAR_THRESHOLD ** 2
            or body.angular_velocity.length_squared() > _WAKE_ANGULAR_THRESHOLD ** 2)


class World:
    def __init__(self):
        self.bodies: list[RigidBody] = []
        self.gravity = Vector3(0, -9.81, 0)
        self.last_contacts: list[ContactManifold] = []

    def add_body(self, body: RigidBody):
        self.bodies.append(body)

    def step(self, dt=1/60):
        for body in self.bodies:
            if body.is_static or body.is_sleeping:
                continue

            body.apply_force(self.gravity * body.mass)

        for body in self.bodies:
            if body.is_sleeping:
                continue
            body.integrate_velocity(dt)
            body.clear_accumulators()

        manifolds: list[ContactManifold] = []
        for body_a, body_b in find_collision_pairs(self.bodies):
            # 衝突判定と衝突解決の処理をここに追加する
            manifold = collide(body_a, body_b)

            if manifold is not None:
                manifolds.append(manifold)
                # 十分速い物体が触れたときだけ起こす。起こさない場合は
                # スリープ側の inv_mass が 0 なので、ソルバーは壁として扱う
                if body_a.is_sleeping and _is_disturbing(body_b):
                    body_a.wake()
                if body_b.is_sleeping and _is_disturbing(body_a):
                    body_b.wake()

        resolve_velocities(manifolds)

        for body in self.bodies:
            if not body.is_sleeping:
                body.integrate_position(dt)

        self._update_sleep_state(dt)
        self.last_contacts = manifolds

    def _update_sleep_state(self, dt: float) -> None:
        for body in self.bodies:
            if body.is_static or body.is_sleeping or not body.can_sleep:
                continue
            if (body.linear_velocity.length_squared() < _SLEEP_LINEAR_THRESHOLD ** 2
                    and body.angular_velocity.length_squared() < _SLEEP_ANGULAR_THRESHOLD ** 2):
                body.sleep_timer += dt
                if body.sleep_timer >= _SLEEP_TIME_REQUIRED:
                    body.sleep()
            else:
                body.sleep_timer = 0.0
