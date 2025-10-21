import os
import time
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import mujoco

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

# ===================== CONFIG =====================
SCENE_PATH   = "scene_and_arm_config/mjx_scene.xml"
ARM_JOINTS   = [f"joint{i}" for i in range(1, 8)]
EE_SITE      = "gripper"        # end-effector site name in your XML
GRIPPER_ACT  = "actuator8"      # actuator controlling finger(s)

# Batch / recording
RECORD_VIDEO   = True
VIDEO_FPS      = 60
VIDEO_SIZE     = (720, 1280)    # (H, W)
RUNS_DIR       = "runs"
HEADLESS       = True           # batch mode (no viewer)

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
ORI_WEIGHT   = 0.01        # reduce orientation pressure vs position
ORI_TOL_DEG  = 2.0         # accept <= 2° tilt

# Pick/Place waypoints
APPROACH_Z   = 0.12        # lift height before lateral motion
BIN_LOCATION = np.array([0.55, -0.35, 0.20])  # target bin center (with clearance)

# Colors & shapes expected in scene names like "obj_red_cube"
AVAILABLE_COLORS = ["red", "blue", "green"]
AVAILABLE_SHAPES = ["sphere", "cube"]

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

def close_gripper(model, data, hold=False, ramp_steps=100, sync_cb=None):
    aid, CLOSE_CMD, OPEN_CMD = gripper_ctrl_values(model)
    if ramp_steps and sync_cb is not None:
        start = float(data.ctrl[aid])
        for i in range(ramp_steps):
            a = (i + 1) / ramp_steps
            data.ctrl[aid] = (1 - a)*start + a*CLOSE_CMD
            mujoco.mj_step(model, data)
            sync_cb()
    data.ctrl[aid] = CLOSE_CMD
    return aid, CLOSE_CMD if hold else None

def apply_arm_targets_via_actuators(model, data, arm_index, arm_act_ids, q_targets_7, preserve_gripper=True):
    """
    Writes joint targets to arm position actuators only (like data.ctrl[:7] = next).
    """
    grip_cmd = None
    gid = None
    if preserve_gripper:
        try:
            gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
            grip_cmd = data.ctrl[gid]
        except RuntimeError:
            grip_cmd = None

    mask = (arm_act_ids >= 0)
    if np.any(mask):
        data.ctrl[arm_act_ids[mask]] = q_targets_7[mask]

    if grip_cmd is not None and gid is not None:
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

# ===================== METRICS & VIDEO ======================
class RunLogger:
    """Collects per-step metrics and saves CSV/NPZ."""
    def __init__(self, model, arm_index, run_dir: Path, site_id: int):
        self.dt        = float(model.opt.timestep)
        self.arm_idx   = arm_index
        self.site_id   = site_id
        self.run_dir   = run_dir
        # Buffers
        self.times     = []
        self.qpos_arm  = []
        self.qvel_arm  = []
        self.ctrl_arm  = []
        self.gripper_u = []
        self.ee_xyz    = []
        self.ee_err    = []
        self.obj_xyz   = []
        self.bin_xyz   = []
        # state
        self.t         = 0.0

    def log_step(self, model, data, goal_xyz=None, obj_body_name=None):
        qpos7 = data.qpos[self.arm_idx.qpos_idx].copy()
        qvel7 = data.qvel[self.arm_idx.qpos_idx].copy()

        ctrl = np.zeros(7)
        take = min(7, model.nu)
        ctrl[:take] = data.ctrl[:take].copy()

        try:
            gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
            grip = float(data.ctrl[gid])
        except RuntimeError:
            grip = np.nan

        ee = data.site_xpos[self.site_id].copy()
        err = float(np.linalg.norm((goal_xyz - ee))) if goal_xyz is not None else np.nan

        objp = np.full(3, np.nan)
        if obj_body_name is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)
            if bid != -1:
                g_first = model.body_geomadr[bid]
                g_count = model.body_geomnum[bid]
                if g_count == 0:
                    objp = data.xipos[bid].copy()
                else:
                    s = np.zeros(3)
                    for gi in range(g_first, g_first + g_count):
                        s += data.geom_xpos[gi]
                    objp = s / g_count

        self.times.append(self.t)
        self.qpos_arm.append(qpos7)
        self.qvel_arm.append(qvel7)
        self.ctrl_arm.append(ctrl)
        self.gripper_u.append(grip)
        self.ee_xyz.append(ee)
        self.ee_err.append(err)
        self.obj_xyz.append(objp)
        self.bin_xyz.append(BIN_LOCATION.copy())

        self.t += self.dt

    def save(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        header = (
            "t,"
            + ",".join([f"q{i+1}" for i in range(7)]) + ","
            + ",".join([f"qd{i+1}" for i in range(7)]) + ","
            + ",".join([f"u{i+1}" for i in range(7)]) + ","
            + "ugrip,"
            + "ee_x,ee_y,ee_z,ee_err,"
            + "obj_x,obj_y,obj_z,"
            + "bin_x,bin_y,bin_z"
        )
        arr = np.column_stack([
            np.array(self.times),
            np.array(self.qpos_arm),
            np.array(self.qvel_arm),
            np.array(self.ctrl_arm),
            np.array(self.gripper_u),
            np.array(self.ee_xyz),
            np.array(self.ee_err),
            np.array(self.obj_xyz),
            np.array(self.bin_xyz),
        ])
        np.savetxt(self.run_dir / "metrics.csv", arr, delimiter=",", header=header, comments="")
        np.savez(self.run_dir / "metrics.npz",
                 t=np.array(self.times),
                 qpos=np.array(self.qpos_arm),
                 qvel=np.array(self.qvel_arm),
                 u_arm=np.array(self.ctrl_arm),
                 u_grip=np.array(self.gripper_u),
                 ee_xyz=np.array(self.ee_xyz),
                 ee_err=np.array(self.ee_err),
                 obj_xyz=np.array(self.obj_xyz),
                 bin_xyz=np.array(self.bin_xyz))

class VideoRecorder:
    """
    Robust off-screen renderer for MuJoCo.
    """

    def __init__(self, model, size=(720, 1280)):
        import numpy as _np
        import sys as _sys
        import threading as _threading
        import mujoco as _mj

        self._np = _np
        self._mj = _mj
        self.frames = []
        self.ok = False

        # GL context handles
        self.renderer = None
        self.cam = None
        self.ctx = None
        self._glfw = None
        self._glfw_win = None

        # Optional media output
        self._media = None
        try:
            import mediapy as _media
            self._media = _media
        except Exception:
            self._media = None

        H, W = int(size[0]), int(size[1])

        # --- Ensure framebuffer large enough ---
        try:
            vis_global = getattr(model.vis, "global")
            ow, oh = int(vis_global.offwidth), int(vis_global.offheight)
        except AttributeError:
            vis_global = None
            ow, oh = 640, 480

        need_resize = (W > ow) or (H > oh)
        if need_resize and vis_global is not None:
            try:
                vis_global.offwidth  = max(W, ow, 64)
                vis_global.offheight = max(H, oh, 64)
            except Exception as e:
                print(f"[VideoRecorder] Could not resize offscreen buffer: {e}")

        try:
            vis_global = getattr(model.vis, "global")
            ow, oh = int(vis_global.offwidth), int(vis_global.offheight)
        except Exception:
            ow, oh = 640, 480

        if W > ow or H > oh:
            print(f"[VideoRecorder] Clamping render size from {H}x{W} to framebuffer {oh}x{ow}")
            H, W = min(H, oh), min(W, ow)

        # --- Create GL context ---
        def _try_mujoco_context():
            for ctor in (lambda: _mj.GLContext(), lambda: _mj.GLContext(offscreen=True)):
                try:
                    ctx = ctor()
                    ctx.make_current()
                    return ctx
                except TypeError:
                    continue
                except Exception:
                    continue
            return None

        def _try_glfw_window(width, height):
            try:
                if _sys.platform == "darwin" and _threading.current_thread() is not _threading.main_thread():
                    return None, None
            except Exception:
                pass

            try:
                import glfw as _glfw
            except Exception as e:
                print(f"[VideoRecorder] glfw not available: {e}")
                return None, None

            if not _glfw.init():
                print("[VideoRecorder] glfw.init() failed")
                return None, None

            _glfw.window_hint(_glfw.VISIBLE, _glfw.FALSE)
            _glfw.window_hint(_glfw.CONTEXT_VERSION_MAJOR, 3)
            _glfw.window_hint(_glfw.CONTEXT_VERSION_MINOR, 3)
            _glfw.window_hint(_glfw.OPENGL_PROFILE, _glfw.OPENGL_CORE_PROFILE)
            try:
                _glfw.window_hint(_glfw.OPENGL_FORWARD_COMPAT, _glfw.TRUE)
            except Exception:
                pass

            win = _glfw.create_window(width, height, "hidden", None, None)
            if not win:
                _glfw.terminate()
                print("[VideoRecorder] glfw.create_window() failed")
                return None, None

            _glfw.make_context_current(win)
            return _glfw, win

        try:
            self.ctx = _try_mujoco_context()
            if self.ctx is None:
                self._glfw, self._glfw_win = _try_glfw_window(W, H)
                if self._glfw_win is None:
                    raise RuntimeError("Could not create any GL context (EGL/OSMesa/GLFW/main-thread).")

            # --- Renderer and camera setup ---
            self.renderer = _mj.Renderer(model, height=H, width=W)

            self.cam = _mj.MjvCamera()
            _mj.mjv_defaultFreeCamera(model, self.cam)

            # 📸 Zoomed-out and slightly top-down
            self.cam.distance = 2       # farther back (was 1.25)
            self.cam.azimuth  = 135       # diagonal view
            self.cam.elevation = -20      # slightly lower for top-down feel
            self.cam.lookat[:] = _np.array([0.4, 0.0, 0.45])

            self.ok = True

        except Exception as e:
            print(f"[VideoRecorder] Off-screen init failed, disabling video. Reason: {e}")
            self._cleanup(partial=True)

    def _cleanup(self, partial=False):
        try:
            if self.renderer is not None:
                close_fn = getattr(self.renderer, "close", None)
                if callable(close_fn):
                    try: close_fn()
                    except Exception: pass
        finally:
            self.renderer = None

        try:
            if self.ctx is not None:
                free_fn = getattr(self.ctx, "free", None)
                if callable(free_fn):
                    try: free_fn()
                    except Exception: pass
        finally:
            self.ctx = None

        try:
            if self._glfw is not None and self._glfw_win is not None:
                self._glfw.destroy_window(self._glfw_win)
                self._glfw.terminate()
        except Exception:
            pass
        finally:
            self._glfw = None
            self._glfw_win = None

        if partial:
            self.ok = False

    def capture(self, data):
        if not self.ok:
            return
        self.renderer.update_scene(data, self.cam)
        img = self.renderer.render()
        self.frames.append(img)

    def save(self, path, fps=60):
        if not self.ok or not self.frames:
            self._cleanup()
            return
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(p)

        try:
            if self._media is not None:
                self._media.write_video(out_path, self.frames, fps=fps)
            else:
                import imageio.v3 as iio
                iio.imwrite(out_path, self._np.stack(self.frames), fps=fps)
        except Exception as e:
            print(f"[VideoRecorder] Could not save video: {e}")
        finally:
            self._cleanup()

# ===================== OBJECT SELECTION ======================
def list_objects(model) -> list[str]:
    names = []
    for bid in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if nm and nm.startswith("obj_"):
            names.append(nm)
    return names

def filter_objects_by_color(objs: Iterable[str], color: str) -> list[str]:
    color = color.lower()
    return [o for o in objs if f"obj_{color}_" in o]

def filter_objects_by_shape(objs: Iterable[str], shape: str) -> list[str]:
    shape = shape.lower()
    return [o for o in objs if o.endswith(f"_{shape}")]

def choose_object(model, color: Optional[str]=None, shape: Optional[str]=None) -> str:
    objs = list_objects(model)
    candidates = objs
    if color:
        candidates = filter_objects_by_color(candidates, color)
    if shape:
        candidates = filter_objects_by_shape(candidates, shape)
    if not candidates:
        raise RuntimeError(f"No objects found for color={color} shape={shape}")
    return random.choice(candidates)

# ===================== TASK HELPERS ======================
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

def ee_pos(model, data, site_id):
    return data.site_xpos[site_id].copy()

def ee_err(goal_xyz, model, data, site_id):
    return np.linalg.norm(goal_xyz - ee_pos(model, data, site_id))

def step_record(model, data, steps, logger: RunLogger, vrec: VideoRecorder|None,
                goal_xyz=None, obj_name=None, hold_grip_cmd=None):
    """Step physics for N steps, logging & optionally capturing frames."""
    gid = None
    if hold_grip_cmd is not None:
        try:
            gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
        except RuntimeError:
            gid = None

    for _ in range(steps):
        if gid is not None:
            data.ctrl[gid] = hold_grip_cmd
        if logger:
            logger.log_step(model, data, goal_xyz=goal_xyz, obj_body_name=obj_name)
        if vrec:
            vrec.capture(data)
        mujoco.mj_step(model, data)

def go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids,
                            goal_xyz, logger: RunLogger, vrec: VideoRecorder|None,
                            max_cycles=6, cycle_steps=120, settle_steps=SETTLE_STEPS, tol=TOL,
                            obj_name=None, hold_grip_cmd=None):
    """
    Closed-loop: repeatedly run IK from CURRENT state -> goal, write actuator setpoints,
    then step physics. Logs metrics & captures video.
    """
    try:
        gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
        grip_cmd = data.ctrl[gid] if hold_grip_cmd is None else hold_grip_cmd
    except RuntimeError:
        gid, grip_cmd = None, None

    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    for cycle in range(max_cycles):
        q_targets = ik.solve_to_joint_targets(goal_xyz, data.qpos.copy())
        apply_arm_targets_via_actuators(model, data, arm_index, arm_act_ids, q_targets)

        steps = cycle_steps if cycle < max_cycles - 1 else settle_steps
        step_record(model, data, steps, logger, vrec,
                    goal_xyz=goal_xyz, obj_name=obj_name, hold_grip_cmd=grip_cmd)

        err = ee_err(goal_xyz, model, data, site_id)
        if err < tol:
            break

def pick_and_place_object(model, data, ik, arm_index, arm_act_ids, logger: RunLogger,
                          vrec: VideoRecorder|None, obj_name: str):
    # Locate object
    obj_xyz  = get_object_coords(model, data, obj_name)

    # Approach above object
    above = obj_xyz.copy(); above[2] += APPROACH_Z
    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, above,
                            logger, vrec, obj_name=obj_name)

    # Open, descend, close (record throughout)
    open_gripper(model, data)
    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, obj_xyz,
                            logger, vrec, obj_name=obj_name)

    _, hold_val = close_gripper(model, data, hold=True, ramp_steps=120,
                                sync_cb=(lambda: None))  # no viewer to sync

    # Hold closed, settle contact
    step_record(model, data, SETTLE_STEPS, logger, vrec,
                goal_xyz=obj_xyz, obj_name=obj_name, hold_grip_cmd=hold_val)

    # Lift and move to bin (hold closed all along)
    lift = obj_xyz.copy(); lift[2] += APPROACH_Z
    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, lift,
                            logger, vrec, obj_name=obj_name, hold_grip_cmd=hold_val)

    go_to_xyz_via_actuators(model, data, ik, arm_index, arm_act_ids, BIN_LOCATION,
                            logger, vrec, obj_name=obj_name, hold_grip_cmd=hold_val)

    # Release + final settle
    open_gripper(model, data)
    step_record(model, data, SETTLE_STEPS, logger, vrec,
                goal_xyz=BIN_LOCATION, obj_name=obj_name)

# ===================== DEMO RUNNER ======================
def run_single_demo(run_dir: Path,
                    kind: str,                 # "single", "color", or "shape"
                    color: Optional[str] = None,
                    shape: Optional[str] = None,
                    seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data  = mujoco.MjData(model)

    # Camera-independent (off-screen video class owns its own camera)
    vrec = VideoRecorder(model, size=VIDEO_SIZE) if RECORD_VIDEO else None

    # Arm indexing + actuator map (exclude gripper)
    arm_index = build_arm_indexing(model)
    try:
        gid = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gid) or None
        exclude = [gname] if gname else None
    except RuntimeError:
        exclude = None
        gid = None
    ARM_ACT_IDS = arm_position_actuator_ids_in_joint_order(model, arm_index, exclude_act_names=exclude)

    # Home the arm by actuators
    home = np.array([0.0, 0.7, 0.0, -1.0, 0.0, 3.0, 0.8])
    mask = (ARM_ACT_IDS >= 0)
    if np.any(mask):
        data.ctrl[ARM_ACT_IDS[mask]] = home[mask]

    # Let it settle and start logging
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    logger = RunLogger(model, arm_index, run_dir, site_id)
    step_record(model, data, SETTLE_STEPS, logger, vrec)

    # IK solver
    ik = GradientDescentIK(model, data, arm_index, site_id,
                           alpha=ALPHA, tol=TOL, max_iters=MAX_ITERS, step_clip=STEP_CLIP,
                           target_z=TARGET_Z, ori_weight=ORI_WEIGHT, ori_tol_deg=ORI_TOL_DEG)

    # Choose object for this demo based on kind/color/shape
    if kind == "single":
        obj_name = choose_object(model)
        tag = "single_random"
    elif kind == "color":
        assert color is not None
        obj_name = choose_object(model, color=color)
        tag = f"color_{color}"
    elif kind == "shape":
        assert shape is not None
        obj_name = choose_object(model, shape=shape)
        tag = f"shape_{shape}"
    else:
        raise ValueError(f"Unknown demo type: {kind}")

    # Perform pick & place
    pick_and_place_object(model, data, ik, arm_index, ARM_ACT_IDS, logger, vrec, obj_name=obj_name)

    # Save artifacts
    logger.save()
    if vrec:
        vrec.save(run_dir / "video.mp4", fps=VIDEO_FPS)

    # Write a tiny summary text
    (run_dir / "meta.txt").write_text(
        f"type={kind}\n"
        f"object={obj_name}\n"
        f"seed={seed}\n"
    )

    print(f"[Saved] {run_dir}")

# ===================== MAIN ======================
def main():
    if not os.path.exists(SCENE_PATH):
        raise FileNotFoundError(f"Scene XML not found: {SCENE_PATH}")

    # Plan: 6 single, 3 color, 2 shape = 11 demos
    plan = []
    # 6 single random
    for i in range(6):
        plan.append(("single", None, None))
    # 3 color-specific (cycle through available colors)
    colors_cycle = (AVAILABLE_COLORS * 2)[:3]
    for c in colors_cycle:
        plan.append(("color", c, None))
    # 2 shape-specific (sphere, cube)
    shapes_cycle = (AVAILABLE_SHAPES * 2)[:2]
    for s in shapes_cycle:
        plan.append(("shape", None, s))

    run_root = Path(RUNS_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    # Save the whole plan
    (run_root / "plan.txt").write_text(
        "\n".join([f"{i+1:03d}: {k} color={c} shape={s}" for i, (k,c,s) in enumerate(plan)])
    )

    # Execute demos
    for i, (kind, color, shape) in enumerate(plan, start=1):
        demo_dir = run_root / f"demo_{i:03d}_{kind}{('_'+color) if color else ''}{('_'+shape) if shape else ''}"
        # Different seeds for variety but deterministic per index
        run_single_demo(demo_dir, kind=kind, color=color, shape=shape, seed=1337 + i)

    print(f"\nAll demos complete. Output in:\n{run_root.resolve()}")

if __name__ == "__main__":
    main()
