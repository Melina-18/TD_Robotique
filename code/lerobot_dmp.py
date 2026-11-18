# DMP (Dynamic Movement Primitives) control of the SO100 arm, in MuJoCo and optionally on the real arm.
#
# State controlled by the DMP (8 dims): [Rotation (rad), gpos x, y, z, roll, pitch, yaw, Jaw (rad)]
# gpos is the gripper pose in the arm plane given by lerobot_FK, the base rotation and the jaw are joints.
#
# Usage:
#   mjpython examples/lerobot_dmp.py                          # built-in demo, played in simulation
#   mjpython examples/lerobot_dmp.py --goal 0.30 0.06         # same movement, new goal (x, z in meters)
#   mjpython examples/lerobot_dmp.py --speed 2                # twice as fast
#   mjpython examples/lerobot_dmp.py --shape infinity         # infinity sign (horizontal 8)
#   mjpython examples/lerobot_dmp.py --alpha 40 --beta 10     # stiffer DMP (alpha_z, beta_z)
#   python   examples/lerobot_dmp.py --record demo.npy        # record a demo by moving the real arm by hand
#   mjpython examples/lerobot_dmp.py --demo demo.npy --real   # learn the recorded demo and play it on the real arm

import argparse
import time

import numpy as np

from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
XML_PATH = "./examples/scene.xml"
PORT = "/dev/tty.usbmodem5B421341721"
CALIBRATION_FILE = "examples/main_follower.json"
DT = 0.01  # DMP integration and control period (s)

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
# Same as the keyboard examples, except x min lowered to 0.10 so that the start pose (x = 0.111) is reachable
control_glimit = [[0.100, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23, 2.0,  1.57,  1.5]]


class DMP:
    """Discrete multi-dimensional DMP, Pastor et al. 2009 formulation (robust when goal == start):

        tau * dv = K (g - y) - D v - K (g - y0) s + K f(s)
        tau * dy = v
        tau * ds = -alpha_s s
        f(s)     = sum_i(psi_i(s) w_i) / sum_i(psi_i(s)) * s

    With the Ijspeert notation tau * dv = alpha_z (beta_z (g - y) - v) + ..., K = alpha_z * beta_z and D = alpha_z.
    beta_z = alpha_z / 4 gives a critically damped system (no oscillation around the goal).
    """

    def __init__(self, n_dims, n_bfs=50, alpha_z=25.0, beta_z=None, alpha_s=4.0):
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.alpha_z = alpha_z
        self.beta_z = alpha_z / 4.0 if beta_z is None else beta_z
        self.K = self.alpha_z * self.beta_z
        self.D = self.alpha_z
        self.alpha_s = alpha_s
        # Basis functions evenly spaced in time, i.e. exponentially spaced in s
        self.centers = np.exp(-alpha_s * np.linspace(0, 1, n_bfs))
        self.widths = n_bfs ** 1.5 / self.centers / alpha_s
        self.w = np.zeros((n_dims, n_bfs))

    def _psi(self, s):
        return np.exp(-self.widths * (s - self.centers) ** 2)

    def fit(self, y_demo, dt):
        """Learn the weights from a demonstration of shape (T, n_dims) sampled every dt seconds."""
        T = len(y_demo)
        self.tau = (T - 1) * dt
        self.y0 = y_demo[0].copy()
        self.g = y_demo[-1].copy()

        yd = np.gradient(y_demo, dt, axis=0)
        ydd = np.gradient(yd, dt, axis=0)
        v, dv = self.tau * yd, self.tau * ydd

        s = np.exp(-self.alpha_s * np.linspace(0, 1, T))
        f_target = (self.tau * dv + self.D * v) / self.K - (self.g - y_demo) + (self.g - self.y0) * s[:, None]

        # Global least squares (ridge) on the normalized basis functions, all dimensions at once
        psi = np.array([self._psi(si) for si in s])  # (T, n_bfs)
        phi = psi / (psi.sum(axis=1, keepdims=True) + 1e-10) * s[:, None]
        self.w = np.linalg.solve(phi.T @ phi + 1e-6 * np.eye(self.n_bfs), phi.T @ f_target).T
        return self

    def rollout(self, y0=None, g=None, tau=None, dt=DT, settle=0.5):
        """Generate a trajectory (T, n_dims) from y0 to g lasting tau seconds (+ settle time to converge on g)."""
        y0 = self.y0 if y0 is None else np.asarray(y0, dtype=float)
        g = self.g if g is None else np.asarray(g, dtype=float)
        tau = self.tau if tau is None else tau

        y, v, s = y0.copy(), np.zeros(self.n_dims), 1.0
        traj = [y.copy()]
        for _ in range(int(round(tau * (1 + settle) / dt))):
            psi = self._psi(s)
            f = self.w @ psi / (psi.sum() + 1e-10) * s
            dv = (self.K * (g - y) - self.D * v - self.K * (g - y0) * s + self.K * f) / tau
            v = v + dv * dt
            y = y + v / tau * dt
            s = s - self.alpha_s * s / tau * dt
            traj.append(y.copy())
        return np.array(traj)


def min_jerk(a, b, t):
    """Smooth interpolation from a to b, t in [0, 1]."""
    t = np.clip(t, 0, 1)
    return a + (b - a) * (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)


def builtin_demo(robot, duration=4.0):
    """Pick-like movement: reach forward over an obstacle, open then close the jaw, turn the base."""
    init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
    t = np.linspace(0, 1, int(round(duration / DT)) + 1)
    demo = np.tile(np.concatenate(([init_qpos[0]], init_gpos, [init_qpos[5]])), (len(t), 1))
    demo[:, 0] = min_jerk(0.0, 0.6, t)                                  # base rotation
    demo[:, 1] = min_jerk(init_gpos[0], 0.25, t)                        # x forward
    demo[:, 3] = min_jerk(init_gpos[2], 0.08, t) + 0.08 * np.sin(np.pi * t) ** 2  # z: go over then down
    demo[:, 7] = -0.157 + 1.2 * np.sin(np.pi * t) ** 2                  # jaw opens then closes
    return demo


def infinity_demo(robot, duration=6.0, center_x=0.20, center_z=0.12, width=0.5, height=0.04):
    """Infinity sign (lemniscate of Gerono) drawn by the gripper in front of the robot:
    horizontal = base rotation (rad), vertical = z (m). Starts and ends at the center, at rest."""
    init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
    t = np.linspace(0, 1, int(round(duration / DT)) + 1)
    theta = 2 * np.pi * min_jerk(0.0, 1.0, t)  # zero velocity at both ends
    demo = np.tile(np.concatenate(([init_qpos[0]], init_gpos, [init_qpos[5]])), (len(t), 1))
    demo[:, 0] = width * np.sin(theta)
    demo[:, 1] = center_x
    demo[:, 3] = center_z + height * np.sin(2 * theta)
    return demo


def state_to_target(state, q_now, robot):
    """Convert a DMP state to joint positions with IK. Returns None if IK fails."""
    gpos = state[1:7].copy()
    gpos[[0, 2]] = np.clip(gpos[[0, 2]], [control_glimit[0][0], control_glimit[0][2]],
                           [control_glimit[1][0], control_glimit[1][2]])
    qpos_inv, ik_success = lerobot_IK(q_now[1:5], gpos, robot=robot)
    if not ik_success:
        return None
    return np.concatenate(([state[0]], qpos_inv[:4], [state[7]]))


def connect_real_arm():
    from lerobot_kinematics import feetech_arm
    arm = feetech_arm(driver_port=PORT, calibration_file=CALIBRATION_FILE)
    # Hold the current pose, then enable torque (it is disabled after calibration)
    arm.arm_hardware.write("Goal_Position", arm.arm_hardware.read("Present_Position"))
    arm.arm_hardware.write("Torque_Enable", 1)
    return arm


def move_smoothly(arm, qpos_target, duration=3.0):
    """Move the real arm from its current pose to qpos_target, starting and stopping gently."""
    start_qpos = arm.feedback()
    for t in np.linspace(0, 1, max(2, int(duration / 0.03))):
        arm.action(min_jerk(start_qpos, qpos_target, t))
        time.sleep(0.03)


def record_demo(path, robot):
    """Kinesthetic teaching: torque off, move the real arm by hand, Ctrl+C to stop."""
    from lerobot_kinematics import feetech_arm
    arm = feetech_arm(driver_port=PORT, calibration_file=CALIBRATION_FILE)
    arm.arm_hardware.write("Torque_Enable", 0)
    print("Torque disabled: move the arm by hand. Press Ctrl+C to stop recording.")
    demo = []
    try:
        while True:
            q = arm.feedback()
            demo.append(np.concatenate(([q[0]], lerobot_FK(q[1:5], robot=robot), [q[5]])))
            time.sleep(DT)
    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect()
    np.save(path, np.array(demo))
    print(f"Saved {len(demo)} samples ({len(demo) * DT:.1f} s) to '{path}'")


def play(traj, robot, real, return_time=8.0, start_time=8.0):
    import mujoco
    import mujoco.viewer

    mjmodel = mujoco.MjModel.from_xml_path(XML_PATH)
    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
    mjdata = mujoco.MjData(mjmodel)
    mjdata.qpos[qpos_indices] = init_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    # Joint trajectory computed once with IK, so the real arm gets a smooth stream of targets
    q_traj, q_now, n_fail = [], init_qpos.copy(), 0
    for state in traj:
        q = state_to_target(state, q_now, robot)
        if q is None:
            n_fail += 1
            q = np.concatenate(([state[0]], q_now[1:5], [state[7]]))
        q_traj.append(q)
        q_now = q
    q_traj = np.array(q_traj)
    print(f"IK failed on {n_fail}/{len(traj)} points (previous joint values kept)")

    arm = None
    if real:
        arm = connect_real_arm()
        move_smoothly(arm, init_qpos)

    # Slow approach from the initial pose to the start of the movement, shown in the simulation too
    approach = np.array([min_jerk(init_qpos, q_traj[0], t)
                         for t in np.linspace(0, 1, max(2, int(start_time / DT)))])

    try:
        with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
            for q in approach:
                if not viewer.is_running():
                    break
                step_start = time.time()
                mjdata.qpos[qpos_indices] = q
                mujoco.mj_forward(mjmodel, mjdata)
                viewer.sync()
                if arm is not None:
                    arm.action(q)
                time_until_next_step = DT - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            while viewer.is_running():
                # Closed shapes (start == end) loop forward, others play forward then backward
                closed = np.allclose(q_traj[0], q_traj[-1], atol=0.05)
                for q in (q_traj if closed else np.concatenate((q_traj, q_traj[::-1]))):
                    if not viewer.is_running():
                        break
                    step_start = time.time()
                    mjdata.qpos[qpos_indices] = q
                    mujoco.mj_forward(mjmodel, mjdata)
                    viewer.sync()
                    if arm is not None:
                        arm.action(q)
                    time_until_next_step = DT - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
    except KeyboardInterrupt:
        print("User interrupted the simulation.")
    finally:
        if arm is not None:
            print(f"Returning slowly to the initial pose ({return_time:.0f} s)...")
            move_smoothly(arm, init_qpos, duration=return_time)
            arm.disconnect()


def plot(demo, traj, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Rotation", "x", "y", "z", "roll", "pitch", "yaw", "Jaw"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for d, ax in enumerate(axes.flat):
        ax.plot(np.arange(len(demo)) * DT, demo[:, d], label="demo")
        ax.plot(np.arange(len(traj)) * DT, traj[:, d], "--", label="DMP")
        ax.set_title(labels[d])
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"Plot saved to '{path}'")


def main():
    parser = argparse.ArgumentParser(description="DMP control of the SO100 arm")
    parser.add_argument("--demo", help="demonstration .npy file (T x 8), default: built-in demo")
    parser.add_argument("--shape", choices=["pick", "infinity"], default="pick", help="built-in demo")
    parser.add_argument("--alpha", type=float, default=25.0, help="DMP alpha_z (damping)")
    parser.add_argument("--beta", type=float, help="DMP beta_z (default alpha/4: critically damped)")
    parser.add_argument("--alpha-s", type=float, default=4.0, help="canonical system decay")
    parser.add_argument("--n-bfs", type=int, default=50, help="number of basis functions")
    parser.add_argument("--record", metavar="FILE", help="record a demonstration on the real arm and exit")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Z"), help="new goal for the gripper (m)")
    parser.add_argument("--speed", type=float, default=1.0, help="speed factor (2 = twice as fast)")
    parser.add_argument("--real", action="store_true", help="also send the trajectory to the real arm")
    parser.add_argument("--plot", metavar="FILE", help="save a demo vs DMP plot (png)")
    parser.add_argument("--start-time", type=float, default=8.0,
                        help="duration of the move from the initial pose to the start of the movement (s)")
    parser.add_argument("--return-time", type=float, default=8.0,
                        help="duration of the return to the initial pose when the window is closed (s)")
    args = parser.parse_args()

    robot = get_robot("so100")

    if args.record:
        record_demo(args.record, robot)
        return

    if args.demo:
        demo = np.load(args.demo)
    elif args.shape == "infinity":
        demo = infinity_demo(robot)
    else:
        demo = builtin_demo(robot)
    dmp = DMP(n_dims=demo.shape[1], n_bfs=args.n_bfs, alpha_z=args.alpha, beta_z=args.beta,
              alpha_s=args.alpha_s).fit(demo, DT)
    print(f"alpha_z={dmp.alpha_z}, beta_z={dmp.beta_z}, alpha_s={dmp.alpha_s}, n_bfs={dmp.n_bfs}")

    goal = dmp.g.copy()
    if args.goal:
        goal[1], goal[3] = args.goal
    traj = dmp.rollout(g=goal, tau=dmp.tau / args.speed)
    print(f"DMP learned from {len(demo)} samples, rollout of {len(traj)} points ({len(traj) * DT:.1f} s)")

    if args.plot:
        plot(demo, traj, args.plot)
    play(traj, robot, args.real, args.return_time, args.start_time)


if __name__ == "__main__":
    main()
