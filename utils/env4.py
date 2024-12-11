'''
    This F1Wrapper is for 
        1. usage issue (gymnasium style)
        2. define observation & action space
        3. observation processing (concatenate lidar + other properties)
        
        ->  outputs of step function should have (n_agents, dim) shape
            ex) observations, rewards, terminates, truncates, infos = env.step([agent_1_action, 
                                                                                agent_2_action,
                                                                                ,,,
                                                                                agent_n_action])
'''

from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os 
import pickle

EPS = 1e-8

def unnormalize_speed(value, minimum, maximum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = np.ones_like(value)*temp_a
    temp_b = np.ones_like(value)*temp_b
    return temp_a*value + temp_b


class F1Wrapper(gym.Wrapper):
    def __init__(self, args, maps, render_mode=None) -> None:
        
        self._env = gym.make("f1tenth_gym:f1tenth-v0", 
                            args=args,
                            maps=maps,
                            render_mode=render_mode)
        super().__init__(self._env)
        self.show_centerline = args.show_centerline

        # for control
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(-np.inf*np.ones(self.obs_dim), np.inf*np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)

    def _reset_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = obs_dict['poses_theta'][0]
        
        # frenet coordinate pose
        poses_s, poses_d, poses_phi = 0, 0, 0
        s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
        poses_s = s
        poses_d = ey
        poses_phi = phi
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def _step_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = obs_dict['poses_theta'][0]
        
        poses_s, poses_d, poses_phi = 0, 0, 0
        s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
        poses_s = s
        poses_d = ey
        poses_phi = phi
        
        # assume that vehicle can't complete 1 cycle in single step
        self.delta_s = poses_s - self.position_frenet[0]
        total_track_s = self._env.track.centerline.spline.s[-1]
        if abs(self.delta_s) > total_track_s/2.0 and self.delta_s < 0:
            # if vehicle moved from spline endpoint to spline startpoint
            self.delta_s += total_track_s
                        
        # frenet coordinate pose
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def reset(self, **kwargs):
        self.history = deque(maxlen=10)
        obs_dict, info = self._env.reset(**kwargs)
        if self.show_centerline:
            self._env.unwrapped.add_render_callback(self._env.track.centerline.render_waypoints)
        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        info['obs_dict'] = obs_dict
        return obs, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own reward function.
    def calc_reward(self, obs_dict, action):
        ego_idx = obs_dict['ego_idx']
        collisions = obs_dict['collisions']
        ego_collision = collisions[ego_idx]
        velocity = action[1]  # Desired speed
        # steering_angle = action[0]
        scan = np.array(obs_dict['scans']).reshape(-1)
        max_distance = 10.0
        normalized_scan = np.clip(scan, 0, max_distance) / max_distance
        obs = normalized_scan.reshape(-1, 10).mean(axis=1)

        wall_penalty1 = np.sum(obs < 0.02) * -0.02
        wall_penalty2 = np.sum(obs < 0.01) * -0.02
        wall_penalty = wall_penalty1 + wall_penalty2

        steering_angle = action[0]
        steering_penalty = -(abs(steering_angle)**2) * 0.02

        # For simplicity, we assume values for speed (v_t), heading error (psi), and cross-track distance (d_c)
        v_t = velocity  # Agent's speed at timestep
        v_max = self.max_speed  # Max speed
        psi = self.yaw_frenet  # Heading error (yaw angle)

        d_c = abs(self.position_frenet[1])  # Cross-track distance (lateral position)
        # Cross-track and heading error reward (scaled with velocity)
        r_cth = (v_t / v_max) * np.cos(psi)**2 * np.sign(np.cos(psi)) + self.delta_s
        track_center = 1-d_c

        # Total reward calculation
        total_reward = -2.0 if ego_collision else r_cth + track_center + wall_penalty + steering_penalty

        # Reward dictionary for debugging and inspection
        reward_dict = {
            'r_cth': r_cth,
            'wall_penalty' : wall_penalty,
            'track_center' : track_center,
            'steering_penalty' : steering_penalty,
        }

        return total_reward, reward_dict
    # ======================================================================== #
        
    def step(self, action:np.array):
        '''
            Original env's action : [desired steer, desired speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : ~inf ~ inf

            Changed to:             [desired steer, normalized speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : -1 ~ 1              -> multiplied by self.max_speed
        '''
        _action = action.copy()
        _action[0] *= self.max_steer
        _action[1] = unnormalize_speed(_action[1], self.min_speed, self.max_speed)
        obs_dict, _, terminate, truncate, info = self._env.step(_action)
        self._step_pose(obs_dict)
        obs = self.getObs(obs_dict)
        reward, reward_dict = self.calc_reward(obs_dict, action)
        info['obs_dict'] = obs_dict
        info.update(reward_dict)
        return obs, reward, terminate, truncate, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own observation.
    # def getObs(self, obs_dict, reset=False):
    #     '''
    #         Original observation is dictionary with keys:
    #             === key ====     === shape ===
    #             'ego_idx'       | int
    #             'scans'         | (num_agents, 1080)
    #             'poses_x'       | (num_agents,)
    #             'poses_y'       | (num_agents,)
    #             'poses_theta'   | (num_agents,)
    #             'linear_vels_x' | (num_agents,)
    #             'linear_vels_y' | (num_agents,)
    #             'ang_vels_z'    | (num_agents,)
    #             'collisions'    | (num_agents,)
    #             'lap_times'     | (num_agents,)
    #             'lap_counts'    | (num_agents,)
            
    #         ==> Changed to include only scans
    #     '''
    #     scan = np.array(obs_dict['scans']).reshape(-1)      # raw lidar value
    #     observation = scan

    #     return observation

    # def getObs(self, obs_dict, reset=False):
    #     scan = np.array(obs_dict['scans']).reshape(-1)[180:-180]  # Focus on 180 degrees (720 points)
    #     max_distance = 10.0
    #     normalized_scan = np.clip(scan, 0, max_distance) / max_distance
    #     observation = normalized_scan.reshape(-1, 10).mean(axis=1)  # Downsample to 72 points
    #     return observation

    def getObs(self, obs_dict, reset=False):
        scan = np.array(obs_dict['scans']).reshape(-1)  # Focus on 180 degrees (720 points)

        ###### ONLY FOR TRAINING!#####
        # Get rid of here before sim2real
        noise_amplitude = 0.01  # Max noise amplitude (1 cm)
        noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=scan.shape)
        # Add noise to raw LIDAR scan
        scan = scan + noise
        ##############################

        max_distance = 10.0
        normalized_scan = np.clip(scan, 0, max_distance) / max_distance
        observation = normalized_scan.reshape(-1, 10).mean(axis=1)  # Downsample to 72 points
        return observation


    # ======================================================================== #

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()