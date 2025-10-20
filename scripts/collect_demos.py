import os, json, time, argparse, random
import numpy as np
import mujoco
from mujoco import viewer


# ---------- CONFIG ----------
XML_PATH = "franka_emika_panda/mjx_scene.xml"
CAMERA_NAME = "top_cam"
IMG_SIZE = (256, 256)
MAX_STEPS = 500

COLORS = ["red", "green", "blue"]
SHAPES = ["cube", "sphere",]
BINS = ["left", "right"]

# ---------- UTILS ----------
def get_joint_qpos(model, data):
    """Return 7 arm joints + 2 fingers."""
    return data.qpos[:9].copy()


def render_rgb(model, data):
    """Render RGB image from a camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    renderer = mujoco.Renderer(model, width=IMG_SIZE[0], height=IMG_SIZE[1])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()
    renderer.close()
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def get_bin_target(bin_name):
    """Approx world coordinates of bin centers (rough)."""
    return np.array([0.5, -0.25, 0.45]) if bin_name == "left" else np.array([0.5, 0.25, 0.45])


# ---------- SCRIPTED POLICY ----------
def scripted_pick_place(model, data, target_geom, target_bin, show_viewer=False):
    """Simple scripted routine: reach → close → lift → move → open."""
    q_home = np.deg2rad([0, -45, 0, -90, 0, 90, 0])
    rgb_log, act_log, prop_log, time_log = [], [], [], []

    # --- Viewer optional setup ---
    v = None
    if show_viewer:
        v = viewer.launch_passive(model, data)

    def step_and_record(ctrl):
        data.ctrl[:] = ctrl[:8]
        mujoco.mj_step(model, data)
        rgb_log.append(render_rgb(model, data))
        act_log.append(ctrl.copy())
        prop_log.append(get_joint_qpos(model, data))
        time_log.append(data.time)
        if v:
            v.sync()
            time.sleep(0.002)

    # --- Sequence of motions ---
    ctrl = np.concatenate([q_home, [0, 0]])  # start open
    for _ in range(100): step_and_record(ctrl)

    # Move slightly toward table (toy motion for demo)
    for _ in range(150):
        ctrl[1] -= 0.001
        step_and_record(ctrl)

    # Close gripper
    for _ in range(80):
        ctrl[-2:] = [-0.02, -0.02]
        step_and_record(ctrl)

    # Lift up
    for _ in range(100):
        ctrl[1] += 0.001
        step_and_record(ctrl)

    # Move sideways toward chosen bin
    for _ in range(150):
        delta = 0.0005 if target_bin == "right" else -0.0005
        ctrl[0] += delta
        step_and_record(ctrl)

    # Open gripper (drop)
    for _ in range(80):
        ctrl[-2:] = [0.0, 0.0]
        step_and_record(ctrl)

    if v:
        # hold final frame briefly so you can see result
        t_end = time.time() + 0.5
        while time.time() < t_end and v.is_running():
            v.sync()
            time.sleep(1 / 60)
        v.close()

    return dict(
        images=np.stack(rgb_log),
        actions=np.stack(act_log),
        proprio=np.stack(prop_log),
        sim_time=np.array(time_log)
    )


# ---------- MAIN LOOP ----------
def main(args):
    os.makedirs(args.out, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    for ep in range(args.num_episodes):
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
        bin_choice = random.choice(BINS)
        geom_name = f"obj_{color}_{shape}"

        instruction = f"Pick up the {color} {shape} and place it in the {bin_choice} bin."
        print(f"\n[Episode {ep}] {instruction}")

        mujoco.mj_resetData(model, data)
        data.time = 0.0

        # small random offset for each object
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.startswith("obj_"):
                data.geom_xpos[g, :2] += np.random.uniform(-0.03, 0.03, 2)

        ep_data = scripted_pick_place(model, data, geom_name, bin_choice, show_viewer=args.show_viewer)

        episode = dict(
            images=ep_data["images"].astype(np.uint8),
            instruction=instruction,
            actions=ep_data["actions"].astype(np.float32),
            proprio=ep_data["proprio"].astype(np.float32),
            sim_time=ep_data["sim_time"].astype(np.float32),
            task_metadata=json.dumps({
                "target_color": color,
                "target_shape": shape,
                "target_bin": bin_choice
            })
        )

        out_path = os.path.join(args.out, f"ep_{ep:06d}.npz")
        np.savez_compressed(out_path, **episode)
        print(f"Saved {out_path} ({len(ep_data['images'])} frames)")

    print("\n✅ Done collecting demonstrations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--out", type=str, default="data/demos")
    parser.add_argument("--show_viewer", action="store_true",
                        help="If set, display MuJoCo viewer while collecting.")
    args = parser.parse_args()
    main(args)