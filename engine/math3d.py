"""Vector3 / Quaternion / Matrix3 / Transform: 物理エンジン全体が依存する数学コア。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vector3":
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vector3":
        length = self.length()
        if length < 1e-12:
            return Vector3(0.0, 0.0, 0.0)
        return self / length

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @staticmethod
    def zero() -> "Vector3":
        return Vector3(0.0, 0.0, 0.0)


class Matrix3:
    """行優先(row-major)の3x3行列。主に慣性テンソルの回転に使う。"""

    __slots__ = ("m",)

    def __init__(self, rows: tuple[tuple[float, float, float], ...] | None = None):
        if rows is None:
            rows = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.m = [list(row) for row in rows]

    @staticmethod
    def identity() -> "Matrix3":
        return Matrix3()

    @staticmethod
    def diagonal(x: float, y: float, z: float) -> "Matrix3":
        return Matrix3(((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z)))

    def transpose(self) -> "Matrix3":
        m = self.m
        return Matrix3(
            (
                (m[0][0], m[1][0], m[2][0]),
                (m[0][1], m[1][1], m[2][1]),
                (m[0][2], m[1][2], m[2][2]),
            )
        )

    def __mul__(self, other: "Matrix3 | Vector3") -> "Matrix3 | Vector3":
        if isinstance(other, Vector3):
            m = self.m
            return Vector3(
                m[0][0] * other.x + m[0][1] * other.y + m[0][2] * other.z,
                m[1][0] * other.x + m[1][1] * other.y + m[1][2] * other.z,
                m[2][0] * other.x + m[2][1] * other.y + m[2][2] * other.z,
            )
        a, b = self.m, other.m
        result = [[0.0, 0.0, 0.0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                result[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
        return Matrix3(tuple(tuple(row) for row in result))

    def __repr__(self) -> str:
        return f"Matrix3({self.m})"


@dataclass
class Quaternion:
    """姿勢を表す単位四元数。w が実部、(x, y, z) が虚部。"""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle: float) -> "Quaternion":
        axis = axis.normalized()
        half = angle * 0.5
        s = math.sin(half)
        return Quaternion(math.cos(half), axis.x * s, axis.y * s, axis.z * s)

    def length(self) -> float:
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Quaternion":
        length = self.length()
        if length < 1e-12:
            return Quaternion.identity()
        inv = 1.0 / length
        return Quaternion(self.w * inv, self.x * inv, self.y * inv, self.z * inv)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """四元数の合成(標準的なHamilton積)。(a * b).rotate_vector(v) は
        b.rotate_vector を先に適用し、その結果に a.rotate_vector を適用するのと等価。"""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate_vector(self, v: Vector3) -> Vector3:
        qv = Quaternion(0.0, v.x, v.y, v.z)
        result = self * qv * self.conjugate()
        return Vector3(result.x, result.y, result.z)

    def to_matrix3(self) -> Matrix3:
        w, x, y, z = self.w, self.x, self.y, self.z
        return Matrix3(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            )
        )

    def integrate(self, angular_velocity: Vector3, dt: float) -> "Quaternion":
        """角速度(ワールド座標系, rad/s)で姿勢を dt だけ進め、正規化して返す。"""
        omega = Quaternion(0.0, angular_velocity.x, angular_velocity.y, angular_velocity.z)
        delta = omega * self
        result = Quaternion(
            self.w + delta.w * 0.5 * dt,
            self.x + delta.x * 0.5 * dt,
            self.y + delta.y * 0.5 * dt,
            self.z + delta.z * 0.5 * dt,
        )
        return result.normalized()


@dataclass
class Transform:
    """位置と姿勢の組。剛体や描画対象のワールド変換を表す。"""

    position: Vector3
    orientation: Quaternion

    @staticmethod
    def identity() -> "Transform":
        return Transform(Vector3.zero(), Quaternion.identity())

    def transform_point(self, local_point: Vector3) -> Vector3:
        return self.position + self.orientation.rotate_vector(local_point)

    def transform_direction(self, local_direction: Vector3) -> Vector3:
        return self.orientation.rotate_vector(local_direction)
