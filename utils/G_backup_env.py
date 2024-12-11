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
    # def calc_reward(self, obs_dict, action):
    #     # forward reward
    #     # 현재 경로와 전문가 경로의 차이를 최소화하려는 보상
    #     reward = 0.0
        
    #     # 1. 전문가 데이터 로드
    #     expert_data_path = 'expert_data.pkl'
    #     if os.path.isfile(expert_data_path):
    #         with open(expert_data_path, 'rb') as f:
    #             expert_data = pickle.load(f)
    #             expert_observations = expert_data['observations']
    #             expert_actions = expert_data['actions']
    #     else:
    #         cprint("[Error] Expert data not found.", bold=True, color="red")
    #         return reward  # 전문가 데이터가 없으면 보상은 0

    #     # 2. 현재 에이전트의 위치 (frenet 좌표)
    #     agent_pos = self.position_frenet  # 현재 에이전트의 frenet 좌표 (s, d)
    #     agent_yaw = self.yaw_frenet  # 현재 에이전트의 yaw (각도)
        
    #     # 3. 전문가 경로에서 가까운 위치 찾기
    #     closest_distance = float('inf')
    #     closest_expert_action = None

    #     for expert_obs, expert_action in zip(expert_observations, expert_actions):
    #         expert_pos = expert_obs[:2]  # 전문가 경로의 (s, d) 좌표 (2D)
    #         distance = np.linalg.norm(agent_pos - expert_pos)  # 에이전트와 전문가의 거리 계산
            
    #         if distance < closest_distance:
    #             closest_distance = distance
    #             closest_expert_action = expert_action  # 가장 가까운 전문가의 행동을 저장
        
    #     # 4. 차이가 적을수록 보상 증가 (거리가 가까울수록 보상)
    #     reward -= closest_distance * 10.0  # 거리가 가까운 행동에 대해 더 많은 보상 부여
        
    #     # 5. 행동에 대한 보상: 전문가 경로에 따라 가까운 행동을 추구
    #     # 전문가의 행동 (속도, 조향 각도)과 현재 에이전트의 행동 차이를 계산
    #     if closest_expert_action is not None:
    #         expert_steer, expert_speed = closest_expert_action
    #         agent_steer, agent_speed = action
            
    #         # 행동 차이 계산 (각도와 속도 차이)
    #         steer_diff = np.abs(agent_steer - expert_steer)
    #         speed_diff = np.abs(agent_speed - expert_speed)
            
    #         # 전문가 행동과 차이가 적을수록 보상
    #         reward -= steer_diff * 5.0  # 조향 각도 차이에 따른 페널티
    #         reward -= speed_diff * 2.0  # 속도 차이에 따른 페널티

    #     # 6. 충돌 여부 체크 (충돌시 큰 페널티 부여)
    #     collision_cost = 10.0 if self.collision else 0.0
    #     reward -= collision_cost  # 충돌시 큰 패널티 부여
        
    #     # 최종 보상 반환
    #     reward_dict = {'forward_reward': reward, 'collision_cost': collision_cost}
    #     return reward, reward_dict
                
    
    def calc_reward(self, obs_dict, action):
        """
        Calculate the reward based on the observation dictionary and action.

        :param obs_dict: Dictionary containing observations.
        :param action: Action taken by the agent (steering, speed).
        :return: Tuple (reward, reward_dict) containing the total reward and detailed reward components.
        """
        # Extract ego vehicle data
        ego_idx = obs_dict['ego_idx']  # Ego vehicle index
        linear_vels_x = obs_dict['linear_vels_x']  # x-direction velocity
        linear_vels_y = obs_dict['linear_vels_y']  # y-direction velocity
        ang_vels_z = obs_dict['ang_vels_z']  # Angular velocity (z-axis)
        collisions = obs_dict['collisions']  # Collision status (1 if collision, else 0)
        scans = np.array(obs_dict['scans']).reshape(-1)  # LiDAR scan data
        lap_times = obs_dict['lap_times']  # Lap times for each agent
        lap_counts = obs_dict['lap_counts']  # Lap counts for each agent

        # Ego information
        ego_vel_x = linear_vels_x[ego_idx]
        ego_vel_y = linear_vels_y[ego_idx]
        ego_collision = collisions[ego_idx]
        lap_time = lap_times[ego_idx]  # Time taken for the last lap
        lap_count = lap_counts[ego_idx]  # Total lap count

        # Action data
        steering_angle = action[0]
        speed = action[1]

        # Reward components
        # 1. Forward progress (maximize forward speed)
        forward_progress = self.delta_s
        # forward_progress = speed
        forward_reward = max(forward_progress, 0)  # Only positive progress is rewarded

        # 2. Steering smoothness penalty (minimize sharp steering)
        # Encourage less aggressive steering adjustments
        steering_penalty = -(abs(steering_angle)**2)* 0.01  # Scale penalty

        # 3. Collision penalty (discourage collisions)
        collision_cost = -50.0 if ego_collision else 0.0

        # 4. Proximity to obstacles (encourage distance from obstacles)
        # Penalize the agent for being too close to obstacles
        min_scan_distance = np.min(scans) if len(scans) > 0 else float('inf')
        obstacle_cost = -5.0 if min_scan_distance < 0.05 else 0.0  # Penalty for being too close


        # 6. Lap time reward (reward fast lap times)
        # If the lap time is lower than a threshold, reward for fast lap
        lap_time_reward = -0.1 * lap_time  # Penalize longer lap times

        # 7. Lap count reward (encourage completing more laps)
        lap_count_reward = lap_count * 2  # Reward for each lap completed

        # # Combine reward components
        # reward = (forward_reward + steering_penalty + collision_cost +
        #         obstacle_cost + speed_penalty + lap_time_reward + lap_count_reward)
        reward = (forward_reward + collision_cost)

        # Detailed reward dictionary for debugging and logging
        # reward_dict = {
        #     'forward_reward': forward_reward,
        #     'steering_penalty': steering_penalty,
        #     'collision_cost': collision_cost,
        #     'obstacle_cost': obstacle_cost,
        #     'speed_penalty': speed_penalty,
        #     'lap_time_reward': lap_time_reward,
        #     'lap_count_reward': lap_count_reward,
        # }
        reward_dict = {
            'forward_reward': forward_reward,
            'collision_cost': collision_cost,
        }

        return reward, reward_dict
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
    
    def getObs(self, obs_dict, reset=False):
        """
        관찰 데이터를 정규화하고 다운샘플링된 LIDAR 스캔으로 구성.
        """
        # 1. LIDAR 데이터 정규화
        scan = np.array(obs_dict['scans']).reshape(-1)  # LIDAR 원시 데이터
        # max_distance = self.max_distance if hasattr(self, 'max_distance') else 10.0  # 최대 거리 기본값 10.0
        # scan = np.clip(scan, 0, max_distance) / max_distance  # 정규화 및 클리핑

        # 2. LIDAR 데이터 다운샘플링
        observation = scan.reshape(-1, 8).mean(axis=1)

        # 3. 관찰값 반환
        return observation

    # ======================================================================== #

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()