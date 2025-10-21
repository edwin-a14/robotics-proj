import os
import time
import random
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer as viewer

# ===================== CONFIG =====================
SCENE_PATH   = "scene_and_arm_config/mjx_scene.xml"
ARM_JOINTS   = [f"joint{i}" for i in range(1, 8)]
EE_SITE      = "gripper"        # end-effector site name in your XML
GRIPPER_ACT  = "actuator8"      # actuator controlling finger(s)

# If your gripper closes at the upper end of ctrlrange, set this True.
FLIP_GRIPPER_LOGIC = False

# IK hyper-parameters
TOL          = 1e-3        # [m] position tolerance
ALPHA        = 0.6         # gradient gain for J^T e
STEP_CLIP    = 0.08        # max |dq| / iter [rad]
MAX_ITERS    = 800
SETTLE_STEPS = 240

# Orientation: keep palm parallel to floor (align local z to world -Z)
TARGET_Z     = np.array([0.0, 0.0, -1.0])  # use [0,0,1] if your tool z points up
ORI_WEIGHT   = 0.01       # ↓ from 1.0 to reduce position-orientation tug-of-war
ORI_TOL_DEG  = 2.0        # accept <= 2° tilt

# Pick/Place waypoints
APPROACH_Z   = 0.12       # lift height before lateral motion
BIN_LOCATION = np.array([0.55, -0.35, 0.20])  # target bin center (with clearance)

# ===================== UTILS ======================
def name_to_id(model, objtype, name: str) -> int:
    _id = mujoco.mj_name2id(model, objtype, name)
    if _id == -1:
        raise RuntimeError(f"{objtype.name} named '{name}' not found")
    return _id

@dataclass
class ArmIndexing:
    jids: list             # joint ids for arm
    qpos_idx: np.ndarray   # qpos indices (len 7)
    dof_idx:  np.ndarray   # dof indices  (len 7)
    limited:  np.ndarray   # which joints have limits
    lo:       np.ndarray   # lower limits
    hi:       np.ndarray   # upper limits

def build_arm_indexing(model) -> ArmIndexing:
    jids, qpos_idx, dof_idx, limited, lo, hi = [], [], [], [], [], []
    for jn in ARM_JOINTS:
        jid = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        jids.append(jid)
        qpos_idx.append(model.jnt_qposadr[jid])
        dof_idx.append(model.jnt_dofadr[jid])
        lim = bool(model.jnt_limited[jid])
        limited.append(lim)
        lo.append(model.jnt_range[jid][0] if lim else -np.inf)
        hi.append(model.jnt_range[jid][1] if lim else +np.inf)
    return ArmIndexing(
        jids=jids,
        qpos_idx=np.array(qpos_idx, dtype=int),
        dof_idx=np.array(dof_idx, dtype=int),
        limited=np.array(limited, dtype=bool),
        lo=np.array(lo, dtype=float),
        hi=np.array(hi, dtype=float),
    )

def clamp_to_limits(qarm: np.ndarray, indexing: ArmIndexing) -> np.ndarray:
    q = qarm.copy()
    mask = indexing.limited
    q[mask] = np.minimum(np.maximum(q[mask], indexing.lo[mask]), indexing.hi[mask])
    return q

def site_R_world(data, site_id: int) -> np.ndarray:
    return np.array(data.site_xmat[site_id]).reshape(3, 3)

def angle_between(u, v):
    c = np.clip(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v)), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def arm_position_actuator_ids_in_joint_order(model, arm_index, exclude_act_names=None):
    """
    1-D array of actuator ids aligned to ARM_JOINTS order; -1 if none for that joint.
    Ensures we only write to the actuators that target those joints (and not the gripper).
    """
    ex = set(exclude_act_names or [])
    ids = []
    for jid in arm_index.jids:
        hit = -1
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a][0] == jid:
                aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
                if aname in ex:
                    continue
                hit = a
                break
        ids.append(hit)
    return np.asarray(ids, dtype=int)

def gripper_ctrl_values(model):
    """
    Returns (aid, CLOSE_CMD, OPEN_CMD) from actuator ctrlrange.
    If FLIP_GRIPPER_LOGIC is True, swap the interpretation.
    """
    aid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
    lo, hi = model.actuator_ctrlrange[aid]
    close_cmd, open_cmd = (lo, hi) if not FLIP_GRIPPER_LOGIC else (hi, lo)
    return aid, float(close_cmd), float(open_cmd)

def open_gripper(model, data, hold=False):
    aid, CLOSE_CMD, OPEN_CMD = gripper_ctrl_values(model)
    data.ctrl[aid] = OPEN_CMD
    return aid, OPEN_CMD if hold else None

def close_gripper(model, data, hold=False, ramp_steps=100, viewer_handle=None):
    aid, CLOSE_CMD, OPEN_CMD = gripper_ctrl_values(model)
    if ramp_steps and viewer_handle is not None:
        start = float(data.ctrl[aid])
        for i in range(ramp_steps):
            a = (i + 1) / ramp_steps
            data.ctrl[aid] = (1 - a)*start + a*CLOSE_CMD
            mujoco.mj_step(model, data)
            viewer_handle.sync()
            time.sleep(0.002)
    data.ctrl[aid] = CLOSE_CMD
    return aid, CLOSE_CMD if hold else None

def apply_arm_targets_via_actuators(model, data, arm_index, arm_act_ids, q_targets_7, preserve_gripper=True):
    """
    Writes joint targets to arm position actuators only (like data.ctrl[:7] = next).
    """
    grip_cmd = None
    if preserve_gripper:
        try:
            gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
            grip_cmd = data.ctrl[gid]
        except RuntimeError:
            grip_cmd = None

    mask = (arm_act_ids >= 0)
    if np.any(mask):
        data.ctrl[arm_act_ids[mask]] = q_targets_7[mask]

    if grip_cmd is not None:
        data.ctrl[gid] = grip_cmd

# ===================== IK SOLVER (J^T e) ======================
class GradientDescentIK:
    """
    Position + 'parallel to floor' orientation IK using Jacobian-transpose.
    Returns joint targets for the 7 arm joints, never commits qpos to the sim.
    """
    def __init__(self, model, data, arm_indexing: ArmIndexing, site_id: int,
                 alpha=ALPHA, tol=TOL, max_iters=MAX_ITERS, step_clip=STEP_CLIP,
                 target_z=TARGET_Z, ori_weight=ORI_WEIGHT, ori_tol_deg=ORI_TOL_DEG):
        self.model = model
        self.data = data
        self.arm = arm_indexing
        self.site_id = site_id
        self.alpha = alpha
        self.tol = tol
        self.max_iters = max_iters
        self.step_clip = step_clip
        self.target_z = target_z / np.linalg.norm(target_z)
        self.ori_weight = ori_weight
        self.ori_tol_deg = ori_tol_deg

        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def _pos_err(self, goal_xyz: np.ndarray) -> np.ndarray:
        x = self.data.site_xpos[self.site_id].copy()
        return goal_xyz - x

    def _ori_err(self) -> tuple[np.ndarray, float]:
        R = site_R_world(self.data, self.site_id)
        z_curr = R[:, 2] / np.linalg.norm(R[:, 2])
        e_r = np.cross(z_curr, self.target_z)   # yaw-free axis alignment error
        ang_deg = angle_between(z_curr, self.target_z)
        return e_r, ang_deg

    def solve_to_joint_targets(self, goal_world_xyz: np.ndarray, qseed_full: np.ndarray) -> np.ndarray:
        """
        Returns a 7-vector of arm joint targets (in ARM_JOINTS order).
        Uses data.qpos only for kinematics, restores it before returning.
        """
        assert qseed_full.shape[0] == self.model.nq

        qpos_save = self.data.qpos.copy()
        self.data.qpos[:] = qseed_full
        mujoco.mj_forward(self.model, self.data)

        q_arm = self.data.qpos[self.arm.qpos_idx].copy()

        for _ in range(self.max_iters):
            e_p = self._pos_err(goal_world_xyz)
            e_r, ang_deg = self._ori_err()

            if np.linalg.norm(e_p) < self.tol and ang_deg <= self.ori_tol_deg:
                break

            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
            Jp = self.jacp[:, self.arm.dof_idx]
            Jr = self.jacr[:, self.arm.dof_idx]

            J_stack = np.vstack((Jp, self.ori_weight * Jr))
            e_stack = np.hstack((e_p, self.ori_weight * e_r))

            dq = self.alpha * (J_stack.T @ e_stack)
            dq = np.clip(dq, -self.step_clip, self.step_clip)

            q_arm = clamp_to_limits(q_arm + dq, self.arm)
            self.data.qpos[self.arm.qpos_idx] = q_arm
            mujoco.mj_forward(self.model, self.data)

        # restore sim state (no teleports)
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)

        return q_arm

# ===================== TASK HELPERS ======================
def choose_object():
    obj_colors = ['red', 'blue', 'green']
    obj_types = ['sphere', 'cube']

    return f"obj_{random.choice(obj_colors)}_{random.choice(obj_types)}"

def get_object_coords(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    # centroid of body's geoms (fallback: body's xipos)
    g_first = model.body_geomadr[bid]
    g_count = model.body_geomnum[bid]
    if g_count == 0:
        return data.xipos[bid].copy()
    pos_sum = np.zeros(3)
    for gi in range(g_first, g_first + g_count):
        pos_sum += data.geom_xpos[gi]
    return pos_sum / g_count

def ee_pos(model, data, site_id):
    return data.site_xpos[site_id].copy()

def ee_err(goal_xyz, model, data, site_id):
    return np.linalg.norm(goal_xyz - ee_pos(model, data, site_id))

def go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids,
                            goal_xyz, settle_steps=SETTLE_STEPS, viewer_handle=None,
                            max_cycles=6, cycle_steps=120, tol=TOL):
    """
    Closed-loop: repeatedly run IK from CURRENT state -> goal, write actuator setpoints,
    then step physics. Stops when EE error < tol or max_cycles is reached.
    """
    # Preserve current gripper command
    try:
        gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
        grip_cmd = data.ctrl[gid]
    except RuntimeError:
        gid, grip_cmd = None, None

    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    for cycle in range(max_cycles):
        # 1) IK from current state → joint targets
        q_targets = ik.solve_to_joint_targets(goal_xyz, data.qpos.copy())

        # 2) Apply targets via actuators (don’t touch gripper)
        apply_arm_targets_via_actuators(model, data, arm_index, arm_act_ids, q_targets)

        # 3) Let the controller pull toward setpoints
        steps = cycle_steps if cycle < max_cycles - 1 else settle_steps
        for _ in range(steps):
            if gid is not None:
                data.ctrl[gid] = grip_cmd  # reassert squeeze/open state
            mujoco.mj_step(model, data)
            if viewer_handle is not None:
                viewer_handle.sync()

        # 4) Check error; stop early if good
        err = ee_err(goal_xyz, model, data, site_id)
        # print(f"[servo] cycle {cycle+1}/{max_cycles} ee_err={err:.4f} m")
        if err < tol:
            break

def pick_up_object(model, data, ik, arm_index, arm_act_ids, viewer_handle=None):
    obj_name = choose_object()
    obj_xyz  = get_object_coords(model, data, obj_name)

    # Approach above object
    above = obj_xyz.copy(); above[2] += APPROACH_Z
    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, above, viewer_handle=viewer_handle)

    # Open, descend, close
    open_gripper(model, data)
    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, obj_xyz, viewer_handle=viewer_handle)
    _, hold_val = close_gripper(model, data, hold=True, ramp_steps=1000, viewer_handle=viewer_handle)
    
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        if viewer_handle is not None:
            viewer_handle.sync()
        time.sleep(0.002)

    # Hold closed, settle contact
    gid, _, _ = gripper_ctrl_values(model)
    for _ in range(SETTLE_STEPS):
        data.ctrl[gid] = hold_val
        mujoco.mj_step(model, data)
        if viewer_handle is not None:
            viewer_handle.sync()
        time.sleep(0.002)

    # Lift and move to bin (hold closed all along)
    lift = obj_xyz.copy(); lift[2] += APPROACH_Z
    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, lift, viewer_handle=viewer_handle)

    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        if viewer_handle is not None:
            viewer_handle.sync()
        time.sleep(0.002)

    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, BIN_LOCATION, viewer_handle=viewer_handle)

    # Optional: release
    open_gripper(model, data)

# ===================== MAIN ======================
def main(scene_path=SCENE_PATH):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    # Camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 1.25
    cam.azimuth  = 135
    cam.elevation = -15
    cam.lookat[:] = np.array([0.4, 0.0, 0.5])

    # Arm indexing + actuator map (exclude gripper)
    arm_index = build_arm_indexing(model)
    try:
        gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gid) or None
        exclude = [gname] if gname else None
    except RuntimeError:
        exclude = None
    ARM_ACT_IDS = arm_position_actuator_ids_in_joint_order(model, arm_index, exclude_act_names=exclude)

    # Sanity prints
    print("ARM_ACT_IDS:", ARM_ACT_IDS, "shape:", ARM_ACT_IDS.shape)
    if gid is not None:
        assert gid not in set(ARM_ACT_IDS[ARM_ACT_IDS >= 0]), "Gripper actuator included in ARM_ACT_IDS!"

    # Home the arm by actuators (no qpos teleports)
    home = np.array([0.0, 0.7, 0.0, -1.0, 0.0, 3.0, 0.8])
    mask = (ARM_ACT_IDS >= 0)
    if np.any(mask):
        data.ctrl[ARM_ACT_IDS[mask]] = home[mask]

    # Let it settle
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    # IK solver (site-based)
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    ik = GradientDescentIK(model, data, arm_index, site_id,
                           alpha=ALPHA, tol=TOL, max_iters=MAX_ITERS, step_clip=STEP_CLIP,
                           target_z=TARGET_Z, ori_weight=ORI_WEIGHT, ori_tol_deg=ORI_TOL_DEG)

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = cam.distance
        v.cam.azimuth = cam.azimuth
        v.cam.elevation = cam.elevation
        v.cam.lookat[:] = cam.lookat[:]

        # Demo: pick and move to bin using actuator-only commands
        pick_up_object(model, data, ik, arm_index, ARM_ACT_IDS, viewer_handle=v)

        print("Press ESC to close viewer.")
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
