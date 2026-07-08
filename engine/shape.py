from dataclasses import dataclass
from .math3d import Vector3, Matrix3


class Shape:

    def local_aabb(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def inertia_tensor(self, mass):
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass
class Box(Shape):
    half_width: float
    half_height: float
    half_depth: float

    def local_aabb(self):
        min_point = Vector3(-self.half_width, -
                            self.half_height, -self.half_depth)
        max_point = Vector3(self.half_width, self.half_height, self.half_depth)
        return (min_point, max_point)

    def inertia_tensor(self, mass):
        Ixx = (1/3) * mass * (self.half_height**2 + self.half_depth**2)
        Iyy = (1/3) * mass * (self.half_width**2 + self.half_depth**2)
        Izz = (1/3) * mass * (self.half_width**2 + self.half_height**2)
        return Matrix3.diagonal(Ixx, Iyy, Izz)


@dataclass
class Sphere(Shape):
    radius: float

    def local_aabb(self):
        r = self.radius
        return (Vector3(-r, -r, -r), Vector3(r, r, r))

    def inertia_tensor(self, mass):
        I = (2.0/5.0) * mass * self.radius**2
        return Matrix3.diagonal(I, I, I)


@dataclass
class Plane(Shape):
    normal: Vector3      # Normal vector of the plane
    offset: float            # Distance from the origin

    def local_aabb(self):
        # A plane is considered to have infinite extent, so we return extreme values.
        # min_point = Vector3(float('-inf'), float('-inf'), float('-inf'))
        # max_point = Vector3(float('inf'), float('inf'), float('inf'))

        big = 1e6
        min_point = Vector3(-big, -big, -big)
        max_point = Vector3(big, big, big)
        return (min_point, max_point)

    def inertia_tensor(self, mass):
        # A plane is considered to have infinite mass and thus an infinite inertia tensor.
        return Matrix3.diagonal(float('inf'), float('inf'), float('inf'))
