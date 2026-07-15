
from dataclasses import dataclass
from itertools import combinations
from typing import List
from .rigid_body import RigidBody
from .math3d import Vector3
from .shape import Box, Sphere, Plane

_PENETRATION_EPSILON = 1e-9  # 貫入深度の閾値。これ以下の貫入は無視する
_PENETRATION_SLOP = 0.005
_POSITION_CORRECTION_PERCENT = 0.2
_PARALLEL_EPSILON = 1e-6


@dataclass
class ContactPoint:
    position: Vector3  # ワールド座標での接触点
    depth: float  # 貫入深度(正の値 = めり込んでいる)


@dataclass
class ContactManifold:
    """2剛体間の接触情報。

    normal の意味: body_a を +normal 方向に、body_b を -normal 方向に
    押すと2剛体が引き離される(貫入が解消する)向きの単位ベクトル。
    """

    body_a: RigidBody
    body_b: RigidBody
    normal: Vector3
    points: list[ContactPoint]


def _swapped(manifold: ContactManifold, body_a: RigidBody, body_b: RigidBody) -> ContactManifold:
    """呼び出し順(body_a, body_b)に合わせて入れ替え、normalの向きを反転する。"""
    return ContactManifold(body_a=body_a, body_b=body_b, normal=-manifold.normal, points=manifold.points)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

# 直方体で囲む1つのAABBを計算する関数
def compute_corner_aabb(body: RigidBody):
    # static/sleeping中は動かないので前回の結果を使い回す(wake時に破棄される)
    if body._aabb_cache is not None:
        return body._aabb_cache

    local_min, local_max = body.shape.local_aabb()
    corners = [
        Vector3(x, y, z)
        for x in (local_min.x, local_max.x)
        for y in (local_min.y, local_max.y)
        for z in (local_min.z, local_max.z)
    ]

    world_corners = [body.direct.rotate_vector(
        corner) + body.position for corner in corners]
    xs = [corner.x for corner in world_corners]
    ys = [corner.y for corner in world_corners]
    zs = [corner.z for corner in world_corners]
    result = (Vector3(min(xs), min(ys), min(zs)),
              Vector3(max(xs), max(ys), max(zs)))
    if body.is_static or body.is_sleeping:
        body._aabb_cache = result
    return result


def _aabb_overlap(a_min: Vector3, a_max: Vector3, b_min: Vector3, b_max: Vector3) -> bool:
    return (
        a_min.x <= b_max.x
        and a_max.x >= b_min.x
        and a_min.y <= b_max.y
        and a_max.y >= b_min.y
        and a_min.z <= b_max.z
        and a_max.z >= b_min.z
    )


def find_collision_pairs(bodies):
    # 動かないもの(static/sleeping)同士のペアは調べない。
    # ほとんどの剛体が静止しているシーンでは、これでペア数が激減する。
    active = [b for b in bodies if not b.is_static and not b.is_sleeping]
    if not active:
        return []
    inactive = [b for b in bodies if b.is_static or b.is_sleeping]

    aabbs = {id(body): compute_corner_aabb(body) for body in bodies}

    collision_pairs = []
    for body_a, body_b in combinations(active, 2):
        if _aabb_overlap(aabbs[id(body_a)][0], aabbs[id(body_a)][1], aabbs[id(body_b)][0], aabbs[id(body_b)][1]):
            collision_pairs.append((body_a, body_b))

    for body_a in active:
        a_min, a_max = aabbs[id(body_a)]
        for body_b in inactive:
            if _aabb_overlap(a_min, a_max, aabbs[id(body_b)][0], aabbs[id(body_b)][1]):
                collision_pairs.append((body_a, body_b))

    return collision_pairs


def collide(body_a: RigidBody, body_b: RigidBody):
    # 衝突判定と衝突解決の処理をここに追加する
    shape_a, shape_b = body_a.shape, body_b.shape

    if isinstance(shape_a, Box) and isinstance(shape_b, Plane):
        return _box_plane_collision(body_a, body_b)

    if isinstance(shape_a, Plane) and isinstance(shape_b, Box):
        manifold = _box_plane_collision(body_b, body_a)
        return None if manifold is None else _swapped(manifold, body_a, body_b)

    if isinstance(shape_a, Box) and isinstance(shape_b, Box):
        # Box同士の衝突判定と解決
        return _box_box_collision(body_a, body_b)

    if isinstance(shape_a, Sphere) and isinstance(shape_b, Plane):
        return _sphere_vs_plane(body_a, body_b)
    if isinstance(shape_a, Plane) and isinstance(shape_b, Sphere):
        manifold = _sphere_vs_plane(body_b, body_a)
        return None if manifold is None else _swapped(manifold, body_a, body_b)

    if isinstance(shape_a, Sphere) and isinstance(shape_b, Box):
        return _sphere_vs_box(body_a, body_b)
    if isinstance(shape_a, Box) and isinstance(shape_b, Sphere):
        manifold = _sphere_vs_box(body_b, body_a)
        return None if manifold is None else _swapped(manifold, body_a, body_b)

    if isinstance(shape_a, Sphere) and isinstance(shape_b, Sphere):
        return _sphere_vs_sphere(body_a, body_b)


    return None


def _box_corners_world(box_body: RigidBody) -> list[Vector3]:
    '''箱の角のワールド座標を返す'''
    box: Box = box_body.shape
    hx, hy, hz = box.half_width, box.half_height, box.half_depth
    signs = (
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    )
    corners = []
    for sx, sy, sz in signs:
        local = Vector3(sx * hx, sy * hy, sz * hz)
        corners.append(box_body.position +
                       box_body.direct.rotate_vector(local))
    return corners


def _box_plane_collision(box_body: RigidBody, plane_body: RigidBody) -> ContactManifold | None:
    # BoxとPlaneの衝突判定と解決の処理をここに追加する
    plane: Plane = plane_body.shape
    box: Box = box_body.shape

    world_normal = plane_body.direct.rotate_vector(plane.normal).normalized()
    world_offset = plane.offset + world_normal.dot(plane_body.position)

    # めり込んでる点
    points: List[ContactPoint] = []

    for corner in _box_corners_world(box_body):
        distance = world_normal.dot(corner)-world_offset
        if distance < _PENETRATION_EPSILON:
            points.append(ContactPoint(position=corner, depth=distance))

    if not points:
        return None

    return ContactManifold(
        body_a=box_body,
        body_b=plane_body,
        normal=world_normal,
        points=points
    )


def _box_axes(body: RigidBody) -> list[Vector3]:
    '''boxの辺の向き'''
    return [
        body.direct.rotate_vector(Vector3(1, 0, 0)),
        body.direct.rotate_vector(Vector3(0, 1, 0)),
        body.direct.rotate_vector(Vector3(0, 0, 1)),
    ]


def _half_extents(body: RigidBody) -> list[float]:
    '''boxのサイズを取り出す'''
    box: Box = body.shape
    return [box.half_width, box.half_height, box.half_depth]

# SAT!!!!!!!!!!!!!!!!!


def _box_box_collision(body_a: RigidBody, body_b: RigidBody):
    # Box同士の衝突判定と解決の処理をここに追加する
    axes_a = _box_axes(body_a)  # 辺
    axes_b = _box_axes(body_b)

    ha = _half_extents(body_a)  # boxのサイズ
    hb = _half_extents(body_b)

    center = body_b.position-body_a.position  # A中心からB中心へのベクトル

    candidates: list[tuple[str, int | None, int | None, Vector3]] = []

    # 15の分離軸を計算
    for i in range(3):
        candidates.append(("faceA", i, None, axes_a[i]))
    for j in range(3):
        candidates.append(("faceB", None, j, axes_b[j]))

    for i in range(3):
        for j in range(3):
            axis = axes_a[i].cross(axes_b[j])
            length = axis.length()
            if length < _PARALLEL_EPSILON:
                # 平行な時は入れなくてよい
                continue
            candidates.append(("edge", i, j, axis/length))

    best: tuple[float, str, int | None, int | None, Vector3] | None = None
    for kind, i, j, axis in candidates:
        ra = sum(ha[k] * abs(axes_a[k].dot(axis)) for k in range(3))
        rb = sum(hb[k] * abs(axes_b[k].dot(axis)) for k in range(3))
        signed_distance = center.dot(axis)
        overlap = ra+rb-abs(signed_distance)

        if overlap < 0:
            # 一つの軸でもOKなら分離している
            return None

        if best is None or overlap < best[0]:
            normal = axis if signed_distance <= 0 else -axis
            best = (overlap, kind, i, j, normal)

    overlap, kind, i, j, normal = best

    if kind == "edge":
        return _edge_edge_manifold(body_a, body_b, axes_a, axes_b, ha, hb, i, j, normal, overlap)

    return _face_manifold(body_a, body_b, axes_a, axes_b, ha, hb, kind, i, j, normal, overlap)


def _edge_point(center: Vector3, axes: list[Vector3], half_extents: list[float], main_index: int, towards: Vector3) -> Vector3:
    point = center
    for k in range(3):
        if k == main_index:
            continue
        sign = 1.0 if axes[k].dot(towards) >= 0 else -1.0
        point = point + axes[k] * (sign * half_extents[k])
    return point


def _closest_points_on_segments(
    p1: Vector3, d1: Vector3, ext1: float, p2: Vector3, d2: Vector3, ext2: float
) -> tuple[Vector3, Vector3]:
    r = p1 - p2
    f = d2.dot(r)
    c_ = d1.dot(r)
    b = d1.dot(d2)
    denom = 1.0 - b * b
    if abs(denom) > 1e-9:
        s = _clamp((b * f - c_) / denom, -ext1, ext1)
    else:
        s = 0.0
    t = b * s + f
    if t < -ext2:
        t = -ext2
        s = _clamp(b * t - c_, -ext1, ext1)
    elif t > ext2:
        t = ext2
        s = _clamp(b * t - c_, -ext1, ext1)
    return p1 + d1 * s, p2 + d2 * t


def _edge_edge_manifold(
    body_a: RigidBody,
    body_b: RigidBody,
    axes_a: list[Vector3],
    axes_b: list[Vector3],
    ha: list[float],
    hb: list[float],
    i: int,
    j: int,
    normal: Vector3,
    overlap: float,
) -> ContactManifold:
    c = body_b.position - body_a.position
    pa = _edge_point(body_a.position, axes_a, ha, i, c)
    pb = _edge_point(body_b.position, axes_b, hb, j, -c)
    point1, point2 = _closest_points_on_segments(
        pa, axes_a[i], ha[i], pb, axes_b[j], hb[j])
    contact_position = (point1 + point2) * 0.5
    return ContactManifold(
        body_a=body_a,
        body_b=body_b,
        normal=normal,
        points=[ContactPoint(position=contact_position, depth=overlap)],
    )


def _face_manifold(
    body_a: RigidBody,
    body_b: RigidBody,
    axes_a: list[Vector3],
    axes_b: list[Vector3],
    ha: list[float],
    hb: list[float],
    kind: str,
    i: int | None,
    j: int | None,
    normal: Vector3,
    overlap: float,
) -> ContactManifold | None:
    if kind == "faceA":
        ref_body, ref_axes, ref_half, ref_index = body_a, axes_a, ha, i
        ref_normal = -normal  # normal は body_a を+へ押すと分離する向きなので、参照面の外向き法線は逆
        inc_body, inc_axes, inc_half = body_b, axes_b, hb
    else:
        ref_body, ref_axes, ref_half, ref_index = body_b, axes_b, hb, j
        ref_normal = normal
        inc_body, inc_axes, inc_half = body_a, axes_a, ha

    ref_sign = 1.0 if ref_axes[ref_index].dot(ref_normal) > 0 else -1.0
    ref_face_center = ref_body.position + \
        ref_axes[ref_index] * (ref_sign * ref_half[ref_index])
    ref_j1, ref_j2 = (k for k in range(3) if k != ref_index)

    inc_index, inc_dot = 0, inc_axes[0].dot(ref_normal)
    for k in (1, 2):
        d = inc_axes[k].dot(ref_normal)
        if abs(d) > abs(inc_dot):
            inc_index, inc_dot = k, d
    inc_sign = -1.0 if inc_dot > 0 else 1.0

    polygon = _face_vertices(inc_body, inc_axes, inc_half, inc_index, inc_sign)

    side_planes = (
        (ref_axes[ref_j1], ref_face_center +
         ref_axes[ref_j1] * ref_half[ref_j1]),
        (-ref_axes[ref_j1], ref_face_center -
         ref_axes[ref_j1] * ref_half[ref_j1]),
        (ref_axes[ref_j2], ref_face_center +
         ref_axes[ref_j2] * ref_half[ref_j2]),
        (-ref_axes[ref_j2], ref_face_center -
         ref_axes[ref_j2] * ref_half[ref_j2]),
    )
    for plane_normal, plane_point in side_planes:
        polygon = _clip_polygon(polygon, plane_normal, plane_point)
        if not polygon:
            break

    points: list[ContactPoint] = []
    for vertex in polygon:
        depth = ref_normal.dot(ref_face_center - vertex)
        if depth > -1e-6:
            points.append(ContactPoint(position=vertex, depth=max(depth, 0.0)))

    if not points:
        return None

    return ContactManifold(body_a=body_a, body_b=body_b, normal=normal, points=points)


def _clip_polygon(polygon: list[Vector3], plane_normal: Vector3, plane_point: Vector3) -> list[Vector3]:
    """Sutherland-Hodgman法: polygon を半空間 (p-plane_point)・normal <= 0 でクリップする。"""
    if not polygon:
        return polygon
    output: list[Vector3] = []
    count = len(polygon)
    for idx in range(count):
        current = polygon[idx]
        previous = polygon[idx - 1]
        current_inside = (current - plane_point).dot(plane_normal) <= 1e-9
        previous_inside = (previous - plane_point).dot(plane_normal) <= 1e-9
        if current_inside:
            if not previous_inside:
                output.append(_intersect_segment_plane(
                    previous, current, plane_normal, plane_point))
            output.append(current)
        elif previous_inside:
            output.append(_intersect_segment_plane(
                previous, current, plane_normal, plane_point))
    return output


def _face_vertices(
    body: RigidBody, axes: list[Vector3], half_extents: list[float], axis_index: int, sign: float
) -> list[Vector3]:
    j1, j2 = (k for k in range(3) if k != axis_index)
    fixed = axes[axis_index] * (sign * half_extents[axis_index])
    combos = ((-1, -1), (1, -1), (1, 1), (-1, 1))  # 矩形を一周する順序
    return [
        body.position + fixed +
        axes[j1] * (s1 * half_extents[j1]) + axes[j2] * (s2 * half_extents[j2])
        for s1, s2 in combos
    ]


def _intersect_segment_plane(p1: Vector3, p2: Vector3, plane_normal: Vector3, plane_point: Vector3) -> Vector3:
    d1 = (p1 - plane_point).dot(plane_normal)
    d2 = (p2 - plane_point).dot(plane_normal)
    t = d1 / (d1 - d2)
    return p1 + (p2 - p1) * t


def _sphere_vs_plane(sphere_body: RigidBody, plane_body: RigidBody) -> ContactManifold | None:
    sphere: Sphere = sphere_body.shape
    plane: Plane = plane_body.shape
    world_normal = plane_body.direct.rotate_vector(plane.normal).normalized()
    world_offset = plane.offset + world_normal.dot(plane_body.position)

    distance = world_normal.dot(sphere_body.position) - world_offset
    depth = sphere.radius - distance
    if depth < -_PENETRATION_EPSILON:
        return None

    contact_position = sphere_body.position - world_normal * sphere.radius
    return ContactManifold(
        body_a=sphere_body,
        body_b=plane_body,
        normal=world_normal,
        points=[ContactPoint(position=contact_position, depth=depth)],
    )




def _sphere_vs_box(sphere_body: RigidBody, box_body: RigidBody) -> ContactManifold | None:
    sphere: Sphere = sphere_body.shape
    axes = _box_axes(box_body)
    half = _half_extents(box_body)

    rel = sphere_body.position - box_body.position
    local = (rel.dot(axes[0]), rel.dot(axes[1]), rel.dot(axes[2]))
    clamped = [_clamp(local[k], -half[k], half[k]) for k in range(3)]
    closest_world = box_body.position + axes[0] * clamped[0] + axes[1] * clamped[1] + axes[2] * clamped[2]

    diff = sphere_body.position - closest_world
    distance = diff.length()
    depth = sphere.radius - distance
    if depth < -_PENETRATION_EPSILON:
        return None

    if distance > 1e-8:
        normal = diff * (1.0 / distance)
    else:
        # 球の中心が箱の内部にある稀なケース: 最も浅い面をnormalに採用する。
        best_index, best_penetration = 0, half[0] - abs(local[0])
        for k in (1, 2):
            penetration = half[k] - abs(local[k])
            if penetration < best_penetration:
                best_index, best_penetration = k, penetration
        sign = 1.0 if local[best_index] >= 0 else -1.0
        normal = axes[best_index] * sign

    return ContactManifold(
        body_a=sphere_body,
        body_b=box_body,
        normal=normal,
        points=[ContactPoint(position=closest_world, depth=depth)],
    )


def _sphere_vs_sphere(body_a: RigidBody, body_b: RigidBody) -> ContactManifold | None:
    sphere_a: Sphere = body_a.shape
    sphere_b: Sphere = body_b.shape

    diff = body_b.position - body_a.position  # a中心からb中心へ
    distance = diff.length()
    depth = sphere_a.radius + sphere_b.radius - distance
    if depth < -_PENETRATION_EPSILON:
        return None

    # normal は body_a を+方向へ押すと分離する向き = bから離れる向き
    normal = (diff * (-1.0 / distance)) if distance > 1e-8 else Vector3(0.0, 1.0, 0.0)
    contact_position = body_a.position - normal * sphere_a.radius

    return ContactManifold(
        body_a=body_a,
        body_b=body_b,
        normal=normal,
        points=[ContactPoint(position=contact_position, depth=depth)],
    )


def correct_penetration(manifolds: list[ContactManifold]) -> None:
    """速度解決後に残る貫入を、位置を直接補正して解消する(疑似速度は使わない簡易版)。"""
    for manifold in manifolds:
        body_a, body_b = manifold.body_a, manifold.body_b
        inv_mass_sum = body_a.inv_mass + body_b.inv_mass
        if inv_mass_sum <= 0:
            continue
        max_depth = max(point.depth for point in manifold.points)
        magnitude = max(max_depth - _PENETRATION_SLOP, 0.0) * \
            _POSITION_CORRECTION_PERCENT / inv_mass_sum
        if magnitude <= 0:
            continue
        correction = manifold.normal * magnitude
        body_a.position = body_a.position + correction * body_a.inv_mass
        body_b.position = body_b.position - correction * body_b.inv_mass
