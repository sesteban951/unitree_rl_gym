import time

import mujoco.viewer
import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
# from isaaclab import LEGGED_GYM_ROOT_DIR
import torch
import yaml

# overwrite the directory that LEGGED_GYM_ROOT_DIR points to
LEGGED_GYM_ROOT_DIR = "/home/sergio/projects/unitree_rl_gym"

# for using the joystick instead of the keyboard
import pygame

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    # get the joystick axis limits
    vx_min = config["joystick"]["vx_min"]
    vx_max = config["joystick"]["vx_max"]
    vy_min = config["joystick"]["vy_min"]
    vy_max = config["joystick"]["vy_max"]
    omega_z_min = config["joystick"]["omega_min"]
    omega_z_max = config["joystick"]["omega_max"]
    deadzone = config["joystick"]["deadzone"]

    # initialize the joystick
    pygame.init()
    pygame.joystick.init()
    num_joysticks = pygame.joystick.get_count()
    if num_joysticks == 0:
        print("No joystick found")
        joystick_availble = False
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_availble = True

    # get the intial simulation time
    t0 = time.time()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # update the joystick
            if joystick_availble==True:
                
                # unpack the raw values
                pygame.event.pump()
                vx = -joystick.get_axis(1)
                vy = -joystick.get_axis(0)
                omega = -joystick.get_axis(3)

                # scale the values to the desired range
                if abs(vx) < deadzone:
                    vx = 0
                else:
                    vx = ((vx_max - vx_min) / 2) * vx
                if abs(vy) < deadzone:
                    vy = 0
                else:
                    vy = ((vy_max - vy_min) / 2) * vy
                if abs(omega) < deadzone:
                    omega_z = 0
                else:
                    omega_z = ((omega_z_max - omega_z_min) / 2) * omega

                cmd = [vx, vy, omega_z]
                print(f"[{vx:.2f}, {vy:.2f}, {omega_z:.2f}]")

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # total obersations (47)
                obs[:3] = omega                  # angular velocity (3)
                obs[3:6] = gravity_orientation   # base orientation (3)
                obs[6:9] = cmd * cmd_scale       # velocity command, vx, vy, angular velocity (3)
                obs[9 : 9 + num_actions] = qj    # current joint positions (12)
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj         # current joint velocities (12)
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action  # last action taken (12)
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase]) # command phase (2)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # print the current time
            # print(time.time()- t0)

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
