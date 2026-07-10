"""cannon_pyramid_scene.py のシミュレーションをGIFとして書き出すスクリプト。

ウィンドウを表示しつつ、フレームをオフスクリーンで読み取って
Pillowでアニメーションgifとして保存する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyray as rl
from PIL import Image

from engine.world import World
from engine.render import Renderer
from engine.rigid_body import RigidBody
from engine.shape import Plane, Box
from engine.math3d import Vector3, Quaternion

OUTPUT_PATH = Path(__file__).resolve().parent / "cannon_pyramid.gif"
SIM_SECONDS = 6.0
DT = 1 / 60
CAPTURE_EVERY = 3  # 60fpsの物理更新のうち何ステップに1回キャプチャするか(=gifのfps)
GIF_SCALE = 0.4    # ファイルサイズ抑制のため縮小


def capture_frame() -> Image.Image:
    img = rl.load_image_from_screen()
    width, height = img.width, img.height
    buf = bytes(rl.ffi.buffer(img.data, width * height * 4))
    rl.unload_image(img)
    frame = Image.frombytes("RGBA", (width, height), buf).convert("RGB")
    if GIF_SCALE != 1.0:
        frame = frame.resize(
            (int(width * GIF_SCALE), int(height * GIF_SCALE)), Image.LANCZOS)
    return frame


def build_scene() -> World:
    world = World()

    ground = RigidBody(Plane(Vector3(0, 1, 0), 0), Vector3(
        0, 0, 0), Quaternion.identity(), 0.0, static=True)
    world.add_body(ground)

    # ピラミッド状に箱を積む
    box_half = 0.5
    rows = 6
    for row in range(rows):
        count = rows - row
        y = box_half + row * box_half * 2
        row_offset = -(count - 1) * box_half
        for i in range(count):
            x = row_offset + i * box_half * 2
            box = RigidBody(Box(box_half, box_half, box_half), Vector3(
                x, y, 0), Quaternion.identity(), 1.0)
            world.add_body(box)

    # 大砲弾(重い箱)を横から高速で撃ち込んでピラミッドを崩す
    cannonball = RigidBody(Box(0.6, 0.6, 0.6), Vector3(-10, box_half, 0),
                            Quaternion.identity(), 8.0)
    cannonball.linear_velocity = Vector3(30, 0, 0)
    world.add_body(cannonball)

    return world


if __name__ == "__main__":
    world = build_scene()
    renderer = Renderer(1000, 700)

    # 崩れる様子全体が収まる固定カメラアングル
    renderer.target = Vector3(0, 2.0, 0)
    renderer.distance = 20.0
    renderer.yaw_deg = -60.0
    renderer.pitch_deg = 18.0
    renderer._update_camera_position()

    # 最初の1枚はバッファのスワップが完了しておらず真っ黒になるため、
    # キャプチャ開始前に数フレーム描画してウォームアップする。
    for _ in range(3):
        renderer.draw_frame(world.bodies)

    frames: list[Image.Image] = []
    total_steps = int(SIM_SECONDS / DT)
    for step in range(total_steps):
        world.step(DT)
        renderer.draw_frame(world.bodies)
        if step % CAPTURE_EVERY == 0:
            frames.append(capture_frame())

    rl.close_window()

    # 全フレームで共通のパレットに減色してファイルサイズを抑える
    palette_source = frames[len(frames) // 2].quantize(colors=48)
    quantized = [f.quantize(palette=palette_source, dither=Image.NONE)
                 for f in frames]

    frame_duration_ms = int(1000 * DT * CAPTURE_EVERY)
    quantized[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=quantized[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames to {OUTPUT_PATH}")
