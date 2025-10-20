import os
import time
import math
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media

# ---------------- CONFIG ----------------
SCENE_PATH   = "scene_and_arm_config/mjx_scene.xml"
ARM_JOINTS   = [f"joint{i}" for i in range(1, 8)]
EE_SITE      = "gripper"        # end-effector site name in your XML
GRIPPER_ACT  = "actuator8"      # actuator controlling finger_joint1

# IK hyper-parameters (tune these)
GOAL_POS     = np.array([0.3, 0.3, 0.1])  # world position target for the EE site
TOL          = 1e-3                           # [m]
ALPHA        = 0.6                            # gradient gain (Jacobian-transpose)
STEP_CLIP    = 0.08                           # max |dq| per-iter (radians)
MAX_ITERS    = 800
SETTLE_STEPS = 300

# ---------------- Utilities ----------------
def name_to_id(model, objtype, name: str) -> int:
    _id = mujoco.mj_name2id(model, objtype, name)
    if _id == -1:
        raise RuntimeError(f"{objtype.name} named '{name}' not found")
    return _id

def open_gripper(model, data):
    aid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
    data.ctrl[aid] = 255

def close_gripper(model, data, viewer_handle=None):
    aid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
    data.ctrl[aid] = 0

    if viewer_handle is not None:
        steps = 100
        for i in range(steps):
            alpha = (i + 1) / steps
            data.ctrl[aid] = 255 + alpha * (0 - 255)
            mujoco.mj_step(model, data)
            viewer_handle.sync()
            time.sleep(0.02)

@dataclass
class ArmIndexing:
    jids: list           # joint ids for arm
    qpos_idx: np.ndarray # qpos indices (len 7)
    dof_idx:  np.ndarray # dof indices  (len 7)
    limited:  np.ndarray # bool array of which joints have limits
    lo:       np.ndarray # lower limits (rad)
    hi:       np.ndarray # upper limits (rad)

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
    # Only clamp joints that are limited
    q = qarm.copy()
    mask = indexing.limited
    q[mask] = np.minimum(np.maximum(q[mask], indexing.lo[mask]), indexing.hi[mask])
    return q

# ---------------- Gradient-Descent IK ----------------
TARGET_Z   = np.array([0.0, 0.0, -1.0])  # use [0,0,1] if your gripper z points up
ORI_WEIGHT = 1.0                          # scales orientation vs position in stacked error
ORI_TOL_DEG = 2.0                         # accept <= 2 degrees tilt from parallel

def site_R_world(data, site_id: int) -> np.ndarray:
    # data.site_xmat is row-major 3x3 in a flat 9-vector
    R = np.array(data.site_xmat[site_id]).reshape(3, 3)
    return R

def angle_between(u, v):
    c = np.clip(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v)), -1.0, 1.0)
    return np.degrees(np.arccos(c))

class GradientDescentIK:
    """
    Position + 'parallel to floor' orientation IK using Jacobian-transpose.
    Enforces: (i) EE site position -> GOAL_POS, (ii) EE local z aligns to TARGET_Z (yaw free).
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

        self.jacp = np.zeros((3, model.nv))  # translational Jacobian
        self.jacr = np.zeros((3, model.nv))  # rotational Jacobian

    def _pos_err(self, goal_xyz: np.ndarray) -> np.ndarray:
        x = self.data.site_xpos[self.site_id].copy()
        return goal_xyz - x

    def _ori_err(self) -> tuple[np.ndarray, float]:
        # Use the site local z-axis in world frame
        R = site_R_world(self.data, self.site_id)
        z_curr = R[:, 2] / np.linalg.norm(R[:, 2])
        # Axis misalignment error (yaw free): z_curr x z_target
        e_r = np.cross(z_curr, self.target_z)
        # Tilt angle (for a clean stop condition)
        ang_deg = angle_between(z_curr, self.target_z)
        return e_r, ang_deg

    def solve(self, goal_world_xyz: np.ndarray, curr: np.ndarray) -> np.ndarray:
        assert curr.shape[0] == self.model.nq

        self.data.qpos[:] = curr
        mujoco.mj_forward(self.model, self.data)

        q_arm = self.data.qpos[self.arm.qpos_idx].copy()

        for _ in range(self.max_iters):
            # Errors
            e_p = self._pos_err(goal_world_xyz)                # (3,)
            e_r, ang_deg = self._ori_err()                     # (3,), scalar

            # Stop if both satisfied
            if np.linalg.norm(e_p) < self.tol and ang_deg <= self.ori_tol_deg:
                break

            # Jacobians at site (world frame)
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)

            # Reduce to arm columns
            Jp = self.jacp[:, self.arm.dof_idx]               # (3,7)
            Jr = self.jacr[:, self.arm.dof_idx]               # (3,7)

            # Stack with orientation weight (yaw remains free via cross-product form)
            J_stack = np.vstack((Jp, self.ori_weight * Jr))   # (6,7)
            e_stack = np.hstack((e_p, self.ori_weight * e_r)) # (6,)

            # Gradient step
            dq = self.alpha * (J_stack.T @ e_stack)
            dq = np.clip(dq, -self.step_clip, self.step_clip)

            # Apply + clamp
            q_arm = clamp_to_limits(q_arm + dq, self.arm)
            self.data.qpos[self.arm.qpos_idx] = q_arm
            mujoco.mj_forward(self.model, self.data)

        return self.data.qpos.copy()
    

def choose_object():
    return 'obj_red_cube'

def get_object_coords(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    g_first = model.body_geomadr[bid]
    g_count = model.body_geomnum[bid]
    if g_count == 0:
        return data.xipos[bid].copy()
    pos_sum = np.zeros(3)
    for gi in range(g_first, g_first + g_count):
        pos_sum += data.geom_xpos[gi]
    return pos_sum / g_count


def move_ee(goal_xyz, curr, model, data, ik, arm_index):
    qpos_result = ik.solve(goal_xyz, curr)

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # Write the IK pose
    data.qpos[:] = qpos_result
    mujoco.mj_forward(model, data)

    print('========')
    print(data.ctrl)
    # If your first 7 actuators are POSITION actuators, make their setpoints = qpos
    data.ctrl[:len(ARM_JOINTS)] = data.qpos[arm_index.qpos_idx]
    print(data.ctrl)

    return qpos_result

def pick_up_object(model, data, ik,arm_index, viewer_handle=None):
    obj_name = choose_object()
    obj_location = get_object_coords(model, data, obj_name)

    open_gripper(model, data)
    move_ee(obj_location, data.qpos.copy(), model, data, ik, arm_index)
    close_gripper(model, data, viewer_handle)

    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        viewer_handle.sync()
        time.sleep(0.002)

    bin_location = np.array([0.55, -0.35, 1])
    
    move_ee(bin_location, data.qpos.copy(), model, data, ik, arm_index)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        viewer_handle.sync()
        time.sleep(0.002)
    open_gripper(model, data)

# ---------------- Main ----------------
def main(scene_path=SCENE_PATH):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    # # Camera (nice wide view)
    # cam = mujoco.MjvCamera()
    # mujoco.mjv_defaultFreeCamera(model, cam)
    # cam.distance = 1.25
    # cam.azimuth  = 135
    # cam.elevation = -15
    # cam.lookat[:] = np.array([0.4, 0.0, 0.5])

    # # Build arm indexing once
    # arm_index = build_arm_indexing(model)

    # # Home pose for the arm (in radians)
    # home = np.array([0.0, 0.7, 0.0, -1.0, 0.0, 3.0, 0.8])
    # # data.qpos[arm_index.qpos_idx] = home
    # data.ctrl[:len(ARM_JOINTS)]   = home  # if position actuators control the joints

    # # settle
    # for _ in range(SETTLE_STEPS):
    #     mujoco.mj_step(model, data)

    # qpos_seed = data.qpos.copy()

    # # IK target & solver
    # site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    # ik = GradientDescentIK(model, data, arm_index, site_id,
    #                        alpha=ALPHA, tol=TOL, max_iters=MAX_ITERS, step_clip=STEP_CLIP)
    
    # # ---- Solve IK ----
    # qpos_result = pick_up_object(model, data, ik, arm_index)

    # # Render before/after for quick visual confirmation
    # renderer = mujoco.Renderer(model)
    # # Before
    # data.qpos[:] = qpos_seed
    # mujoco.mj_forward(model, data)
    # renderer.update_scene(data, cam)
    # img_before = renderer.render()

    # # After
    # data.qpos[:] = qpos_result
    # mujoco.mj_forward(model, data)
    # ee_after = data.site_xpos[site_id].copy()
    # renderer.update_scene(data, cam)
    # img_after = renderer.render()

    # print(f"Goal (world): {GOAL_POS}")
    # print(f"EE after IK : {ee_after}")
    # print(f"Position error (m): {np.linalg.norm(GOAL_POS - ee_after):.6f}")

    # # Optional: open a passive viewer to inspect interactively
    # with viewer.launch_passive(model, data) as v:
    #     v.cam.distance = cam.distance
    #     v.cam.azimuth = cam.azimuth
    #     v.cam.elevation = cam.elevation
    #     v.cam.lookat[:] = cam.lookat[:]

    #     target_ctrl = data.qpos[arm_index.qpos_idx].copy()
    #     while v.is_running():
    #         # Hold at the IK solution if actuators are position-type
    #         data.ctrl[:len(ARM_JOINTS)] = target_ctrl
    #         mujoco.mj_step(model, data)
    #         v.sync()

        # Camera (nice wide view)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 1.25
    cam.azimuth  = 135
    cam.elevation = -15
    cam.lookat[:] = np.array([0.4, 0.0, 0.5])

    # Build arm indexing once
    arm_index = build_arm_indexing(model)

    # Home pose for the arm (in radians)
    home = np.array([0.0, 0.7, 0.0, -1.0, 0.0, 3.0, 0.8])
    # data.qpos[arm_index.qpos_idx] = home
    data.ctrl[:len(ARM_JOINTS)]   = home  # if position actuators control the joints

    # settle
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = cam.distance
        v.cam.azimuth = cam.azimuth
        v.cam.elevation = cam.elevation
        v.cam.lookat[:] = cam.lookat[:]
        qpos_seed = data.qpos.copy()

        # IK target & solver
        site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        ik = GradientDescentIK(model, data, arm_index, site_id,
                            alpha=ALPHA, tol=TOL, max_iters=MAX_ITERS, step_clip=STEP_CLIP)
        
        # ---- Solve IK ----
        pick_up_object(model, data, ik, arm_index, viewer_handle=v)

        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.002)

        

        # # Render before/after for quick visual confirmation
        # renderer = mujoco.Renderer(model)
        # # Before
        # data.qpos[:] = qpos_seed
        # mujoco.mj_forward(model, data)
        # renderer.update_scene(data, cam)
        # img_before = renderer.render()

        # # After
        # data.qpos[:] = qpos_result
        # mujoco.mj_forward(model, data)
        # ee_after = data.site_xpos[site_id].copy()
        # renderer.update_scene(data, cam)
        # img_after = renderer.render()

        # print(f"Goal (world): {GOAL_POS}")
        # print(f"EE after IK : {ee_after}")
        # print(f"Position error (m): {np.linalg.norm(GOAL_POS - ee_after):.6f}")
    

if __name__ == "__main__":
    main()
