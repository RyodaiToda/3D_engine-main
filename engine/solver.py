import math

from .math3d import Vector3
from .collision import ContactManifold


# 反復回数は接触点数に応じて調整する(適応反復)。
# 静止した高い塔を支えるには20回必要だが、それは接触点が少ない平時の話。
# 崩壊中など接触点が多い時は収束精度より速度を優先して10回まで下げる
_MAX_VELOCITY_ITERATIONS = 20
_MIN_VELOCITY_ITERATIONS = 10
_ITERATION_BUDGET = 3000  # 1ステップあたりの「反復x接触点」の上限目安
_RESTITUTION_VELOCITY_THRESHOLD = 1

# Baumgarte安定化: 貫入深度に比例した分離速度を目標に混ぜ、めり込みの蓄積を防ぐ。
# これがないと、積み上げた箱が毎ステップ僅かに沈み込み続け、
# やがてSATの最小貫入軸が水平方向に切り替わって横滑りで崩れる
_BAUMGARTE = 0.1
_PENETRATION_SLOP = 0.005  # この深さまでの貫入は許容する(接触の安定用)


def _point_velocity(body, r: Vector3) -> Vector3:
    '''rの点の、並進と回転を合わせた速度'''
    return body.linear_velocity + body.angular_velocity.cross(r)


def _apply_impulse(body_a, body_b, r_a: Vector3, r_b: Vector3, impulse: Vector3) -> None:
    '''撃力を与えて速度を更新'''
    body_a.linear_velocity = body_a.linear_velocity + impulse * body_a.inv_mass
    body_a.angular_velocity = body_a.angular_velocity + \
        body_a.inv_inertia_world * r_a.cross(impulse)

    body_b.linear_velocity = body_b.linear_velocity - impulse * body_b.inv_mass
    body_b.angular_velocity = body_b.angular_velocity - \
        body_b.inv_inertia_world * r_b.cross(impulse)


def _tangent_basis(normal: Vector3) -> tuple[Vector3, Vector3]:
    '''ベクトルnormalに対して、垂直な2つの正規直交ベクトルを出す'''
    arbitrary = Vector3(1, 0, 0) if abs(normal.x) < 0.9 else Vector3(0, 1, 0)
    tangent1 = normal.cross(arbitrary).normalized()
    tangent2 = normal.cross(tangent1)
    return tangent1, tangent2


def _effective_mass(body_a, body_b, r_a: Vector3, r_b: Vector3, direction: Vector3) -> float:
    angular_a = (body_a.inv_inertia_world *
                 r_a.cross(direction)).cross(r_a).dot(direction)
    angular_b = (body_b.inv_inertia_world *
                 r_b.cross(direction)).cross(r_b).dot(direction)
    return body_a.inv_mass + body_b.inv_mass + angular_a + angular_b


def resolve_velocities(manifolds: list[ContactManifold], dt: float = 1 / 60):
    contacts = []
    for manifold in manifolds:
        contacts.extend(_prepare_contact(manifold, dt))

    if not contacts:
        return

    iterations = max(_MIN_VELOCITY_ITERATIONS,
                     min(_MAX_VELOCITY_ITERATIONS,
                         _ITERATION_BUDGET // len(contacts)))
    for _ in range(iterations):
        for contact in contacts:
            _solve_contact_point(contact)


def _prepare_contact(manifold: ContactManifold, dt: float):
    body_a, body_b = manifold.body_a, manifold.body_b
    normal = manifold.normal

    restitution = math.sqrt(body_a.restitution * body_b.restitution)
    friction = math.sqrt(body_a.friction * body_b.friction)

    tangent1, tangent2 = _tangent_basis(normal)

    prepared = []

    for point in manifold.points:
        r_a = point.position-body_a.position  # Aから見ためり込んでる点
        r_b = point.position-body_b.position
        rel_vel = _point_velocity(body_a, r_a) - \
            _point_velocity(body_b, r_b)  # Bから見たAの相対速度
        approach_speed = rel_vel.dot(normal)  # 接触面方向の速度

        if approach_speed < -_RESTITUTION_VELOCITY_THRESHOLD:
            target_speed = -restitution*approach_speed
        else:
            target_speed = 0.0

        # めり込みが深いほど強く押し戻す(Baumgarte)。反発の目標速度と大きい方を採る
        bias_speed = _BAUMGARTE * max(point.depth - _PENETRATION_SLOP, 0.0) / dt
        target_speed = max(target_speed, bias_speed)

        prepared.append(
            {
                "body_a": body_a,
                "body_b": body_b,
                "normal": normal,
                "tangent1": tangent1,
                "tangent2": tangent2,
                "friction": friction,
                "r_a": r_a,
                "r_b": r_b,
                "k_normal": _effective_mass(body_a, body_b, r_a, r_b, normal),
                "k_t1": _effective_mass(body_a, body_b, r_a, r_b, tangent1),
                "k_t2": _effective_mass(body_a, body_b, r_a, r_b, tangent2),
                "target_speed": target_speed,
                "accum_normal": 0.0,
                "accum_t1": 0.0,
                "accum_t2": 0.0,
            }
        )
    return prepared


# ここが一番重要！！！！！！！！！！！
def _solve_contact_point(data: dict):
    body_a, body_b = data["body_a"], data["body_b"]
    normal = data["normal"]
    r_a, r_b = data["r_a"], data["r_b"]

    # 垂直方向
    if data["k_normal"] > 0:
        rel_vel = _point_velocity(body_a, r_a)-_point_velocity(body_b,r_b)
        vn = rel_vel.dot(normal)  # 相対速度の垂直成分

        # 垂直成分の撃力 targetspeed : 衝突後に、なってほしい速度
        lambda_n = (data["target_speed"]-vn)/data["k_normal"]
        new_accum = max(data["accum_normal"]+lambda_n, 0)
        applied = new_accum - data["accum_normal"]

        data["accum_normal"] = new_accum
        _apply_impulse(body_a, body_b, r_a, r_b, normal*applied)

    # 水平方向1
    max_friction = data["friction"]*data["accum_normal"]
    for tangent, accum_key, k_key in (
        (data["tangent1"], "accum_t1", "k_t1"),
        (data["tangent2"], "accum_t2", "k_t2")
    ):
        if data[k_key] <= 0:
            # 接触と垂直方向の
            continue

        rel_vel = _point_velocity(body_a, r_a)-_point_velocity(body_b, r_b)
        vt = rel_vel.dot(tangent)
        lambda_t = -vt/data[k_key]
        old_accum = data[accum_key]
        new_accum_t = max(-max_friction, min(max_friction, old_accum+lambda_t))
        applied_t = new_accum_t-old_accum
        data[accum_key] = new_accum_t
        _apply_impulse(body_a, body_b, r_a, r_b, tangent*applied_t)
