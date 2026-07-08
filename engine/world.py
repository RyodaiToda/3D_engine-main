from itertools import combinations

from .collision import ContactManifold, find_collision_pairs
from .rigid_body import RigidBody
from .math3d import Vector3, Matrix3
from .collision import collide
from .solver import resolve_velocities


class World:
    def __init__(self):
        self.bodies: list[RigidBody] = []
        self.gravity = Vector3(0, -9.81, 0)
        self.last_contacts: list[ContactManifold] = []

    def add_body(self, body: RigidBody):
        self.bodies.append(body)

    def step(self, dt=1/60):
        for body in self.bodies:
            if body.is_static:
                continue

            body.apply_force(self.gravity * body.mass)

        for body in self.bodies:
            body.integrate_velocity(dt)
            body.clear_accumulators()

        manifolds: list[ContactManifold] = []
        for body_a, body_b in find_collision_pairs(self.bodies):
            # 衝突判定と衝突解決の処理をここに追加する
            manifold = collide(body_a, body_b)

            if manifold is not None:
                manifolds.append(manifold)

        resolve_velocities(manifolds)

        for body in self.bodies:
            body.integrate_position(dt)

        self.last_contacts = manifolds
