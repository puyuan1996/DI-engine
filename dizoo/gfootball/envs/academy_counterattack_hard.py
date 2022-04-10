from dizoo.gfootball.envs.multiagentenv import MultiAgentEnv
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
from ding.utils import ENV_REGISTRY
from typing import Any, List, Union, Optional
import copy
import torch
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.torch_utils import to_ndarray, to_list

@ENV_REGISTRY.register('counter')
class Academy_Counterattack_Hard(MultiAgentEnv):

    def __init__(
        self,
        cfg: dict,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=1000,
        render=False,
        n_agents=4,
        time_limit=150,
        time_step=0,
        obs_dim=34,
        env_name='academy_counterattack_hard',
        stacked=False,
        representation="simple115",
        rewards='scoring',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        # seed=0
    ):
        self._cfg = cfg   # TODO
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.obs_dim = obs_dim
        self.env_name = env_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        # self.seed = seed

        self._env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        # self._env.seed(self.seed)

        obs_space_low = self._env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self._env.observation_space.high[0][:self.obs_dim]

        self._action_space = [gym.spaces.Discrete(
            self._env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self._observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self._env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self._reward_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)  # TODO

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim  # QPLEX unit_dim for cds_gfootball
        # self.unit_dim = 8  # QPLEX unit_dim set like that in Starcraft II

    def get_simple_obs(self, index=-1):
        full_obs = self._env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team']
                              [-self.n_agents:].reshape(-1))
            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

            simple_obs.append(full_obs['right_team'][0])
            simple_obs.append(full_obs['right_team'][1])
            simple_obs.append(full_obs['right_team'][2])
            simple_obs.append(full_obs['right_team_direction'][0])
            simple_obs.append(full_obs['right_team_direction'][1])
            simple_obs.append(full_obs['right_team_direction'][2])

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            simple_obs.append(full_obs['right_team'][0] - ego_position)
            simple_obs.append(full_obs['right_team'][1] - ego_position)
            simple_obs.append(full_obs['right_team'][2] - ego_position)
            simple_obs.append(full_obs['right_team_direction'][0])
            simple_obs.append(full_obs['right_team_direction'][1])
            simple_obs.append(full_obs['right_team_direction'][2])

            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def get_global_special_state(self):
        return [np.concatenate([self.get_global_state(), self.get_obs_agent(i)]) for i in range(self.n_agents)]

    def check_if_done(self):
        cur_obs = self._env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self._env.reset()
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])

        obs = {
            'agent_state': np.stack(self.get_obs(),axis=0).astype(np.float32),
            # 'global_state': self.get_state(),
            'global_state': np.stack(self.get_global_special_state(),axis=0,).astype(np.float32),
            'action_mask': np.stack(self.get_avail_actions(),axis=0).astype(np.float32),
        }
        # obs = to_ndarray(obs).astype(np.float32)
        
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0

        # return obs, self.get_global_state()
        return obs

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        if isinstance(actions, np.ndarray):
            actions=torch.from_numpy(actions)
        
        _, original_rewards, done, infos = self._env.step(actions.to('cpu').numpy().tolist())

        obs = {
            'agent_state': np.stack(self.get_obs(),axis=0).astype(np.float32),
            # 'global_state': self.get_state(),
            'global_state': np.stack(self.get_global_special_state(),axis=0,).astype(np.float32),
            'action_mask': np.stack(self.get_avail_actions(),axis=0).astype(np.float32),
        }
        # obs = to_ndarray(obs).astype(np.float32)


        rewards = list(original_rewards)
        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        if sum(rewards) <= 0:
            # return obs, self.get_global_state(), -int(done), done, infos
            infos['final_eval_reward'] = infos['score_reward'] # TODO
            # return -int(done), done, infos
            return BaseEnvTimestep(obs, -int(done), done, infos)

    # def reset(self):
    #     """Returns initial observations and states."""
    #     self.time_step = 0
    #     self._env.reset()
    #     # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])

    #     obs = {
    #         'agent_state': torch.tensor(np.stack(self.get_obs(),axis=0),dtype=torch.float32),
    #         # 'global_state': self.get_state(),
    #         'global_state': torch.tensor(np.stack(self.get_global_special_state(),axis=0,),dtype=torch.float32),
    #         'action_mask': torch.tensor(np.stack(self.get_avail_actions(),axis=0),dtype=torch.float32),
    #     }
    #     # obs = to_ndarray(obs).astype(np.float32)
        
    #     if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
    #         np_seed = 100 * np.random.randint(1, 1000)
    #         self._env.seed(self._seed + np_seed)
    #     elif hasattr(self, '_seed'):
    #         self._env.seed(self._seed)
    #     self._final_eval_reward = 0

    #     # return obs, self.get_global_state()
    #     return obs

    # def step(self, actions):
    #     """Returns reward, terminated, info."""
    #     self.time_step += 1
    #     if isinstance(actions, np.ndarray):
    #         actions=torch.from_numpy(actions)
    #     _, original_rewards, done, infos = self._env.step(actions.to('cpu').numpy().tolist())

    #     obs = {
    #         'agent_state': torch.tensor(np.stack(self.get_obs(),axis=0),dtype=torch.float32),
    #         # 'global_state': self.get_state(),
    #         'global_state': torch.tensor(np.stack(self.get_global_special_state(),axis=0,),dtype=torch.float32),
    #         'action_mask': torch.tensor(np.stack(self.get_avail_actions(),axis=0),dtype=torch.float32),
    #     }
    #     # obs = to_ndarray(obs).astype(np.float32))

    #     rewards = list(original_rewards)
    #     # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

    #     if self.time_step >= self.episode_limit:
    #         done = True

    #     if self.check_if_done():
    #         done = True

    #     if sum(rewards) <= 0:
    #         # return obs, self.get_global_state(), -int(done), done, infos
    #         infos['final_eval_reward'] = infos['score_reward'] # TODO
    #         # return -int(done), done, infos
    #         return BaseEnvTimestep(obs, torch.tensor(-int(done), dtype=torch.float32), done, infos)

        infos['final_eval_reward'] = infos['score_reward'] # TODO
        # return obs, self.get_global_state(), 100, done, infos
        return BaseEnvTimestep(obs, torch.tensor(100, dtype=torch.float32), done, infos)


    def get_obs(self):
        """Returns all agent observations in a list."""
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    # def reset(self):
    #     """Returns initial observations and states."""
    #     self.time_step = 0
    #     self._env.reset()
    #     obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])

    #     return obs, self.get_global_state()

    def render(self):
        pass

    def close(self):
        self._env.close()

    def save_replay(self):
        """Save a replay."""
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]
