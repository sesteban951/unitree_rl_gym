import time

import mujoco.viewer
import mujoco
import numpy as np
# from isaaclab import LEGGED_GYM_ROOT_DIR
import torch
import yaml

# overwrite the directory that LEGGED_GYM_ROOT_DIR points to
LEGGED_GYM_ROOT_DIR = "/home/sergio/projects/unitree_rl_gym"

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


def translate_mujoco_to_isaac(mujoco_vector, DOF_NUM=23):
    # Initialize the Isaac joint order to zero values
    isaac_vector = np.zeros(DOF_NUM)
    # ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_elbow_roll_joint', 'right_elbow_roll_joint', 'left_five_joint', 'left_three_joint', 'left_zero_joint', 'right_five_joint', 'right_three_joint', 'right_zero_joint', 'left_six_joint', 'left_four_joint', 'left_one_joint', 'right_six_joint', 'right_four_joint', 'right_one_joint', 'left_two_joint', 'right_two_joint']
    # Mapping of Mujoco joint indices to their corresponding indices in the Isaac joint order
    mujoco_to_isaac_indices = {
        0: 0,   # left_hip_pitch_joint
        1: 3,   # left_hip_roll_joint
        2: 7,   # left_hip_yaw_joint
        3: 11,  # left_knee_joint
        4: 15,  # left_ankle_pitch_joint
        5: 19,  # left_ankle_roll_joint
        6: 1,   # right_hip_pitch_joint
        7: 4,   # right_hip_roll_joint
        8: 8,   # right_hip_yaw_joint
        9: 12,  # right_knee_joint
        10: 16, # right_ankle_pitch_joint
        11: 20, # right_ankle_roll_joint
        12: 2,  # torso_joint
        13: 5,  # left_shoulder_pitch_joint
        14: 9,  # left_shoulder_roll_joint
        15: 13, # left_shoulder_yaw_joint
        16: 17, # left_elbow_pitch_joint
        17: 21, # left_elbow_roll_joint
        18: 6,  # right_shoulder_pitch_joint
        19: 10, # right_shoulder_roll_joint
        20: 14, # right_shoulder_yaw_joint
        21: 18, # right_elbow_pitch_joint
        22: 22, # right_elbow_roll_joint
        # 23: 25, # left_zero_joint
        # 24: 31, # left_one_joint
        # 25: 36, # left_two_joint
        # 26: 24, # left_three_joint
        # 27: 30, # left_four_joint
        # 28: 23, # left_five_joint
        # 29: 29, # left_six_joint
        # 30: 26, # right_zero_joint
        # 31: 34, # right_one_joint
        # 32: 35, # right_two_joint
        # 33: 27, # right_three_joint
        # 34: 33, # right_four_joint
        # 35: 26, # right_five_joint
        # 36: 32, # right_six_joint
    }
    
    # Fill the Isaac vector with values from the Mujoco vector based on the mapping
    for mujoco_index, isaac_index in mujoco_to_isaac_indices.items():
        isaac_vector[isaac_index] = mujoco_vector[mujoco_index]
    
    return isaac_vector

def translate_isaac_to_mujoco(isaac_vector, DOF_NUM=23):
    # Initialize the Mujoco joint order
    mujoco_vector = np.zeros(DOF_NUM)
    
    # Mapping of Isaac joint indices to their corresponding indices in the Mujoco joint order
    isaac_to_mujoco_indices = {
        0: 0,   # left_hip_pitch_joint
        3: 1,   # left_hip_roll_joint
        7: 2,   # left_hip_yaw_joint
        11: 3,  # left_knee_joint
        15: 4,  # left_ankle_pitch_joint
        19: 5,  # left_ankle_roll_joint
        1: 6,   # right_hip_pitch_joint
        4: 7,   # right_hip_roll_joint
        8: 8,   # right_hip_yaw_joint
        12: 9,  # right_knee_joint
        16: 10, # right_ankle_pitch_joint
        20: 11, # right_ankle_roll_joint
        2: 12,  # torso_joint
        5: 13,  # left_shoulder_pitch_joint
        9: 14,  # left_shoulder_roll_joint
        13: 15, # left_shoulder_yaw_joint
        17: 16, # left_elbow_pitch_joint
        21: 17, # left_elbow_roll_joint
        6: 18,  # right_shoulder_pitch_joint
        10: 19, # right_shoulder_roll_joint
        14: 20, # right_shoulder_yaw_joint
        18: 21, # right_elbow_pitch_joint
        22: 22, # right_elbow_roll_joint
        # 25: 23, # left_zero_joint
        # 31: 24, # left_one_joint
        # 36: 25, # left_two_joint
        # 24: 26, # left_three_joint
        # 30: 27, # left_four_joint
        # 23: 28, # left_five_joint
        # 29: 29, # left_six_joint
        # 26: 30, # right_zero_joint
        # 34: 31, # right_one_joint
        # 35: 32, # right_two_joint
        # 27: 33, # right_three_joint
        # 33: 34, # right_four_joint
        # 26: 35, # right_five_joint
        # 32: 36, # right_six_joint
        
    }
    
    # Fill the Mujoco vector with values from the Isaac vector based on the mapping
    for isaac_index, mujoco_index in isaac_to_mujoco_indices.items():
        # print("isaac_index: ", isaac_index)
        # print("mujoco_index: ", mujoco_index)
        mujoco_vector[mujoco_index] = isaac_vector[isaac_index]
    
    return mujoco_vector



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

        try:
            isaac_mujoco_conversion = config["isaac_mujoco_conversion"]
        except KeyError:
            isaac_mujoco_conversion = False

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
    # load to cuda
    if torch.cuda.is_available():
        policy = policy.cuda()

    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count < 1:
        print("No joystick detected, using initial command from config instead.")
        joystick = None
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Using controller: {joystick.get_name()}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = 1  # ID of the robot's main body
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
                if joystick is not None:
                    for event in pygame.event.get():
                        pass
                    # Left stick: control vx, vy (2D plane), right stick X-axis: vyaw
                    vy = -(joystick.get_axis(0))
                    vx = -(joystick.get_axis(1))
                    vyaw = -(joystick.get_axis(3))

                    # Clip or zero out small values
                    if abs(vx) < 0.1:
                        vx = 0
                    else:
                        vx = np.clip(vx, -0.5, 0.5)
                    if abs(vy) < 0.1:
                        vy = 0
                    else:
                        vy = np.clip(vy, -0.3, 0.3)
                    if abs(vyaw) < 0.1:
                        vyaw = 0
                    else:
                        vyaw = np.clip(vyaw, -1.5, 1.5)
                    cmd[0] = vx
                    cmd[1] = vy
                    cmd[2] = vyaw 

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                if isaac_mujoco_conversion:
                    qj = translate_mujoco_to_isaac(qj, num_actions)
                    dqj = translate_mujoco_to_isaac(dqj, num_actions)
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                print("cmd: ", cmd)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                if torch.cuda.is_available():
                    obs_tensor = obs_tensor.cuda()
                    action = policy(obs_tensor).detach().cpu().numpy().squeeze()
                else:
                    action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                if isaac_mujoco_conversion:
                    target_dof_pos = translate_isaac_to_mujoco(action, num_actions) * action_scale + default_angles
                    # for i in range(num_actions):
                    #     if i == 12 or i == 13 or i == 14 or i == 15 or i == 16 or i == 17 or i == 18 or i == 19 or i == 20 or i == 21 or i == 22:
                    #         target_dof_pos[i] = default_angles[i]
                else:
                    target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)