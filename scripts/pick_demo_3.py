import os
import numpy as np
import mujoco
import mujoco.viewer as viewer
from dataclasses import dataclass
import time
# ---------------- CONFIG ----------------
SCENE_PATH   = "scene_and_arm_config/mjx_scene.xml"
ARM_JOINTS   = [f"joint{i}" for i in range(1, 8)]
GRIPPER_SITE = "gripper"
GRIPPER_ACT  = "actuator8"     # controls finger_joint1 via equality (mirrors the other finger)
EE_SITE = "gripper"

# Target pose policy: keep tool Z axis pointing DOWN (toward -world Z)
WORLD_Z_DOWN  = np.array([0.0, 0.0, -1.0])

# Tune this if your gripper site isn't exactly at the pinch point.
# Positive values push the site ABOVE the object (since we subtract along local -Z).
TIP_OFFSET_M = -0.03  # ~5.5 cm typical Panda finger pad offset from site
CENTER_EPS   = 2.0e-3  # 2 mm alignment tolerance

def get_body_geom_centroid(model, data, body_name):
    """Return the average (x,y,z) of all geoms attached to a body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    g_first = model.body_geomadr[bid]
    g_count = model.body_geomnum[bid]

    if g_count == 0:
        # Fallback: body COM
        return data.xipos[bid].copy()
    pos_sum = np.zeros(3)
    n = 0
    for gi in range(g_first, g_count + g_first):
        pos_sum += data.geom_xpos[gi]
        n += 1
    
    return pos_sum / max(n, 1)

def make_R_tool_z_down():
    # Z axis straight down in world; X,Y orthonormal from a hint to keep frame stable
    z_down = np.array([0.0, 0.0, -1.0])
    y_hint = np.array([0.0, 1.0, 0.0])
    z = z_down / (np.linalg.norm(z_down) + 1e-9)
    x = np.cross(y_hint, z); x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # columns

# --------- Gripper helpers ---------
def open_gripper(model, data):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
    if aid == -1:
        raise RuntimeError(f"Actuator '{GRIPPER_ACT}' not found.")
    data.ctrl[aid] = 0.40  # open wider than range to fully open via bias

def animate_gripper_to(model, data, target, viewer_handle=None):
    """
    Smoothly animate the gripper control value to `target` so you can watch it close in the viewer.
    Works in MuJoCo v3. If `viewer_handle` is provided, we sync each frame.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
    if aid == -1:
        raise RuntimeError(f"Actuator '{GRIPPER_ACT}' not found.")

    # Clamp to actuator ctrl range if available
    if model.actuator_ctrlrange.size >= 2 * model.nu:
        lo, hi = model.actuator_ctrlrange[aid]
    else:
        lo, hi = 0.0, 0.04

    start = float(np.clip(data.ctrl[aid], lo, hi))
    goal  = float(np.clip(target,      lo, hi))

    steps = 10
    for t in range(steps):
        alpha = (t + 1) / steps
        data.ctrl[aid] = start + alpha * (goal - start)
        mujoco.mj_step(model, data)
        if viewer_handle is not None:
            viewer_handle.sync()
            time.sleep(0.002 * 10)


def grip_to_object_width(model, data, squeeze=0.002,
                         animate=True, viewer_handle=None):
    """
    Set/animate the gripper opening to (estimated_width - squeeze).
    If animate=True, interpolates so you can watch it close in the viewer.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
    if aid == -1:
        raise RuntimeError(f"Actuator '{GRIPPER_ACT}' not found.")

    width = 0.02 # TODO: MAKE DYNAMIC
    target = max(width - float(squeeze), 0.0)

    if animate:
        animate_gripper_to(model, data, target, viewer_handle=viewer_handle)
    else:
        # Non-animated fallback (instant set)
        if model.actuator_ctrlrange.size >= 2 * model.nu:
            lo, hi = model.actuator_ctrlrange[aid]
        else:
            lo, hi = 0.0, 0.04
        data.ctrl[aid] = float(np.clip(target, lo, hi))

# ---------------- Utilities ----------------
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def body_world_pos(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    return data.xpos[bid].copy()

def set_camera_full_view(v):
    v.cam.lookat[:] = np.array([0.75, 0.00, 0.35])
    v.cam.distance  = 2.2
    v.cam.azimuth   = 120
    v.cam.elevation = -25

def site_xmat(model, data, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    m = data.site_xmat[sid]  # row-major, 9 elements
    return m.reshape(3, 3)

def make_R_from_zy(z_axis: np.ndarray, y_hint: np.ndarray = np.array([0.0, 1.0, 0.0])):
    """
    Build a rotation matrix whose z-axis = z_axis (normalized),
    and x/y chosen to make a right-handed frame using y_hint for stability.
    """
    z = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    # x = normalize( y_hint x z ); y = z x x
    x = np.cross(y_hint, z); n = np.linalg.norm(x)
    if n < 1e-8:
        # fallback if y_hint ~ z
        y_hint = np.array([1.0, 0.0, 0.0])
        x = np.cross(y_hint, z); n = np.linalg.norm(x)
    x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)  # columns are axes
    return R

# --------------- IK (6D pose, stable) ---------------
@dataclass
class IKConfig:
    iters: int = 300
    pos_tol: float = 1.0e-3
    rot_tol: float = 1.0e-3
    damping: float = 2.0e-2   # DLS lambda
    max_step: float = 0.05    # max |dq| per joint each iter (rad)
    step_scale: float = 0.8
    w_pos: float = 1.0
    w_rot: float = 0.6
    # small nullspace to home to avoid elbow flips
    nullspace_weight: float = 0.001
    q_home: np.ndarray = None  # set at runtime

def rotation_error_vec(R_current, R_target):
    """
    Classic SO(3) error: phi = 0.5 * sum_i (R_i x R*_i)
    where columns are basis vectors.
    """
    rc0, rc1, rc2 = R_current[:, 0], R_current[:, 1], R_current[:, 2]
    rt0, rt1, rt2 = R_target[:, 0], R_target[:, 1], R_target[:, 2]
    return 0.5 * (np.cross(rc0, rt0) + np.cross(rc1, rt1) + np.cross(rc2, rt2))

def solve_ik_pose(model, data, site_name, p_target, R_target, arm_dofs, cfg: IKConfig):
    """
    6D IK: match (position, orientation). Writes small joint updates into qpos
    each iteration, enforcing joint limits. Returns a copy of the final qpos.
    """
    qpos = data.qpos.copy()
    sid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid == -1:
        raise RuntimeError(f"Site '{site_name}' not found.")

    # Precompute joint info
    dof_info = []
    for nv_idx in arm_dofs:
        jnt_id = model.dof_jntid[nv_idx]
        qadr   = model.jnt_qposadr[jnt_id]
        lo, hi = model.jnt_range[jnt_id]
        dof_info.append((nv_idx, jnt_id, qadr, lo, hi))

    for _ in range(cfg.iters):
        write_arm_qpos_only(model, data, qpos, ARM_JOINTS)
        mujoco.mj_forward(model, data)

        p_cur = data.site_xpos[sid].copy()
        R_cur = data.site_xmat[sid].reshape(3, 3).copy()

        ep = p_target - p_cur
        er = rotation_error_vec(R_cur, R_target)

        if np.linalg.norm(ep) < cfg.pos_tol and np.linalg.norm(er) < cfg.rot_tol:
            break

        # Jacobians at the site
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)

        # Stack pos/rot with weights
        J = np.vstack([cfg.w_pos * jacp[:, arm_dofs],
                       cfg.w_rot * jacr[:, arm_dofs]])  # 6 x 7
        e = np.hstack([cfg.w_pos * ep, cfg.w_rot * er])  # (6,)

        # Damped least squares
        JJt = J @ J.T
        dq_arm = J.T @ np.linalg.solve(JJt + (cfg.damping**2) * np.eye(6), e)
        dq_arm *= cfg.step_scale

        # Nullspace bias toward home (if provided)
        if cfg.q_home is not None:
            q_now = np.array([qpos[info[2]] for info in dof_info])
            dq_ns = cfg.nullspace_weight * (cfg.q_home - q_now)
            dq_arm += dq_ns

        # Clamp per-joint step
        inf_norm = np.linalg.norm(dq_arm, ord=np.inf)
        if inf_norm > cfg.max_step:
            dq_arm *= (cfg.max_step / (inf_norm + 1e-9))

        # Apply and clamp to joint limits
        for k, (_, __, qadr, lo, hi) in enumerate(dof_info):
            qpos[qadr] = clamp(qpos[qadr] + dq_arm[k], lo, hi)

    return qpos

# --------------- Motion primitives ----------------
def set_arm_ctrl_to_qpos(model, data, qpos_target):
    # Position actuators (first 7) follow the joints by setting ctrl (desired angle)
    for i, jn in enumerate(ARM_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = model.jnt_qposadr[jid]
        data.ctrl[i] = qpos_target[qadr]

def write_arm_qpos_only(model, data, qpos_target, arm_joint_names):
    """Copy only the 7 Panda arm joint angles into qpos; leave gripper/etc. untouched."""
    for jn in arm_joint_names:
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = model.jnt_qposadr[jid]   # hinge -> 1 dof
        data.qpos[qadr] = qpos_target[qadr]

def wait_until_site_near(model, data, site_name, target_pos, tol=2e-3, max_steps=2500):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    for _ in range(max_steps):
        mujoco.mj_step(model, data)
        if np.linalg.norm(data.site_xpos[sid] - target_pos) < tol:
            return True
    return False

def move_to_pose(model, data, p_target, R_target, arm_dofs, ik_cfg: IKConfig):
    qpos_target = solve_ik_pose(model, data, EE_SITE, p_target, R_target, arm_dofs, ik_cfg)
    write_arm_qpos_only(model, data, qpos_target, ARM_JOINTS)
    set_arm_ctrl_to_qpos(model, data, qpos_target)
    wait_until_site_near(model, data, EE_SITE, p_target)

# ---------------- High-level pick/place ----------------
def pick_object(model, viewer, data, obj_name, arm_dofs, approach, grasp, lift, tip_offset=TIP_OFFSET_M):
    """
    3 waypoints: above -> grasp -> lift. We keep tool-Z down (vertical).
    """
    # Build desired orientation (Z down), X/Y world-aligned
    R_des = make_R_from_zy(WORLD_Z_DOWN)

    p_obj   = body_world_pos(model, data, obj_name)
    p_above = p_obj.copy(); p_above[2] = max(p_obj[2] + approach, 0.12)

    tip_world_offset = R_des @ np.array([0.0, 0.0, tip_offset])  # equals -tip_offset * world_z
    p_grasp = p_obj.copy(); p_grasp[2] = max(p_obj[2] + grasp,   0.07)
    p_lift  = p_grasp.copy(); p_lift[2] += lift

    # IK config with a mild home nullspace
    home = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 2.3, 0.8])
    ikcfg = IKConfig(q_home=home)

    open_gripper(model, data)
    move_to_pose(model, data, p_above, R_des, arm_dofs, ikcfg)
    move_to_pose(model, data, p_grasp, R_des, arm_dofs, ikcfg)

    # 3) XY micro-alignment loop at constant Z (reduce residual millimeters)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    for _ in range(8):
        mujoco.mj_forward(model, data)
        p_site_now = data.site_xpos[sid].copy()

        # Re-read object center (in case it drifted/rolled)
        p_obj_now = get_body_geom_centroid(model, data, obj_name)
        # Desired site XY so that tip ends at object XY:
        p_site_xy_target = p_obj_now[:2]+ (tip_world_offset[:2])
        delta_xy = p_site_xy_target - p_site_now[:2]

        if np.linalg.norm(delta_xy) < CENTER_EPS:
            break

        # Nudge in XY only; keep Z fixed at current grasp height
        p_nudge = p_site_now.copy()
        p_nudge[:2] += delta_xy * 0.8  # gentle proportional nudge

        move_to_pose(model, data, p_nudge, R_des, arm_dofs, ikcfg)

    grip_to_object_width(model, data, squeeze=0.002, animate=True, viewer_handle=viewer)
    # move_to_pose(model, data, p_lift,  R_des, arm_dofs, ikcfg)

# ---------------- Main ----------------
def main(scene_path=SCENE_PATH):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    # Arm dof indices (nv) for the Panda hinge joints
    arm_dofs = []
    for jn in ARM_JOINTS:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if j_id == -1:
            raise RuntimeError(f"Joint {jn} not found.")
        arm_dofs.append(model.jnt_dofadr[j_id])

    # Home pose
    home = [0.0, -0.7, 0.0, -2.0, 0.0, 2.3, 0.8]
    for i, jn in enumerate(ARM_JOINTS):
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = home[i]
        data.ctrl[i]    = home[i]
    open_gripper(model, data)

    # Let physics settle
    for _ in range(300): mujoco.mj_step(model, data)

    # Find objects named 'obj_*' (MuJoCo v3)
    object_names = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith("obj_"):
            object_names.append(name)
    print("Found objects:", object_names)

    with viewer.launch_passive(model, data) as v:
        set_camera_full_view(v)

        # short idle for camera
        for _ in range(180):
            mujoco.mj_step(model, data); v.sync()

        for obj in object_names:
            print(f"Picking {obj}...")
            pick_object(
                model, v, data, obj, arm_dofs,
                approach=0.16,         
                grasp=0.01,
                lift=0.22     
            )
            for _ in range(300):
                mujoco.mj_step(model, data); v.sync()
            break;

        print("Done. Press ESC to close.")
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    main()
