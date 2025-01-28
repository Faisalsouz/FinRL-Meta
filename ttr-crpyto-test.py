# get more info about this here. https://chatgpt.com/share/6790f22c-24f0-800b-9023-bc1caa8fa0f5

from __future__ import annotations

# from finrl.config import INDICATORS
# from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from my_meta.custom_crypto_env import CryptoTradingEnv
from my_meta.custom_crypto_paper_trading import AlpacaPaperTradingCryptoLive
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
#from finrl.meta.data_processor import DataProcessor
from meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import numpy as np
import pandas as pd
import os
import time
import gym

import numpy.random as rd
import torch
import torch.nn as nn
from copy import deepcopy
from torch import Tensor
from torch.distributions.normal import Normal
import torch
from finrl.config import ERL_PARAMS

# from finrl.config import RLlib_PARAMS
# from finrl.config import SAC_PARAMS
# from finrl.config import TRAIN_END_DATE
# from finrl.config import TRAIN_START_DATE
# from finrl.config_tickers import DOW_30_TICKER

# from finrl.config import RLlib_PARAMS
# from finrl.config import TEST_END_DATE
# from finrl.config import TEST_START_DATE
# from finrl.config_tickers import DOW_30_TICKER
import matplotlib.pyplot as plt
import plotly.express as px
   # Visualize results with Plotly
import plotly.graph_objects as go
from dotenv import load_dotenv
import os











# from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor


CRYPTO_TICKER = ["BTCUSDT"]  # Single asset
ticker_list = CRYPTO_TICKER
INDICATORS = ['macd', 'rsi', 'cci', 'dx']



#end of the imports:


class ActorPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)

        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        self.agent_class = agent_class  # agent = agent_class(...)

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(128)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(2000)  # collect horizon_len step while exploring, then update network
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.eval_times = int(32)  # number of times that get episodic cumulative return
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)


def get_gym_env_args(env, if_print: bool) -> dict:
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]

    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    print(f"env_args = {repr(env_args)}") if if_print else None
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_class=None, env_args=None):
    if env_class.__module__ == 'gym.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.soft_update_tau = args.soft_update_tau

        self.states = None  # assert self.states == (1, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        print("Using device:", self.device)

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), args.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), args.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, horizon_len: int) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.states[0]

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action, logprob = [t.squeeze(0) for t in get_action(state.unsqueeze(0))[:2]]

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, _, _ = env.step(ary_action)
            if done:
                ary_state, _ = env.reset()

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> [float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = torch.tensor(self.states, dtype=torch.float32).to(self.device)
        next_value = self.cri(next_state).detach()[0, 0]

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages


class PendulumEnv(gym.Wrapper):  # a demo of custom gym env
    def __init__(self):
        gym.logger.set_level(40)  # Block warning
        gym_env_name = "Pendulum-v0" if gym.__version__ < '0.18.0' else "Pendulum-v1"
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self) -> np.ndarray:  # reset the agent in env
        resetted_env, _ = self.env.reset()
        return resetted_env

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):  # agent interacts in env
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, done, info_dict, _ = self.env.step(action * 2)
        return state.reshape(self.state_dim), float(reward), done, info_dict

    
def train_agent(args: Config):
    args.init_before_training()

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)

    new_env, _ = env.reset()
    agent.states = new_env[np.newaxis, :]

    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                          eval_per_step=args.eval_per_step,
                          eval_times=args.eval_times,
                          cwd=args.cwd)
    torch.set_grad_enabled(False)
    while True: # start training
        buffer_items = agent.explore_env(env, args.horizon_len)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer_items)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), args.cwd + '/actor.pth')
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")

        
class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = []
        print(f"\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              f"\n| `time`: Time spent from the start of training to this moment."
              f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `avgS`: Average of steps in an episode."
              f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")
            
    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))
        
        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, _ = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    return cumulative_returns, episode_steps + 1

# DRL Agent Class

# from elegantrl.agents import AgentA2C

MODELS = {"ppo": AgentPPO}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
      

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
         
            "if_train": True,
        }
        environment = self.env(config=env_config)
        env_args = {'config': env_config,
              'env_name': environment.env_name,
              'state_dim': environment.state_dim,
              'action_dim': environment.action_dim,
              'if_discrete': False}
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Config(agent_class=agent, env_class=self.env, env_args=env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dims = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent_class = MODELS[model_name]
        environment.env_num = 1
        agent = agent_class(net_dimension, environment.state_dim, environment.action_dim)
        actor = agent.act
        # load agent
        try:  
            cwd = cwd + '/actor.pth'
            print(f"| load actor from: {cwd}")
            actor.load_state_dict(torch.load(cwd, map_location=lambda storage, loc: storage))
            act = actor
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state, _ = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _, _ = environment.step(action)

                total_asset = (
                    environment.amount
                    + (
                        environment.price_ary[environment.day] * environment.stocks
                    ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return episode_total_assets
    
#Test and train functinos: 
def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
   
    **kwargs,
):
    print("Received kwargs:", kwargs)
    # download data
    dp = DataProcessor(data_source,start_date, end_date, time_interval)
    price_array, tech_array,_ = dp.run(ticker_list,
                                                        technical_indicator_list, 
                                                       if_vix=False , cache=True)
    # data = dp.clean_data(data)
    # data = dp.add_technical_indicator(data, technical_indicator_list)
 
    # price_array, tech_array, turbulence_array = dp.df_to_array(data)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        DRLAgent_erl = DRLAgent
        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,

        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
        

def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=False,
    **kwargs,
):
    """Enhanced test function with detailed visualizations"""
    print("Received kwargs:", kwargs)
    
    # 1) Load & process new data with DataProcessor
    dp = DataProcessor(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval
    )
    price_array, tech_array, _ = dp.run(
        ticker_list=ticker_list,
        technical_indicator_list=technical_indicator_list,
        if_vix=if_vix,
        cache=True
    )

    # 2) Build the test environment
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # 3) Load the trained policy
    net_dimension = kwargs.get("net_dimension", [64, 32])
    cwd = kwargs.get("cwd", f"./{model_name}")
    
    if drl_lib == "elegantrl":
        DRLAgent_erl = DRLAgent
        
        # Initialize tracking lists
        episode_total_assets = []
        actions_history = []
        portfolio_allocations = []
        profits_per_step = []
        prices_history = []
        trade_dates = []  # New: Track actual dates
        
        # Get initial state
        state, _ = env_instance.reset()
        initial_asset = env_instance.initial_total_asset
        
        # Setup device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = MODELS[model_name](net_dimension, env_instance.state_dim, env_instance.action_dim)
        agent.act.load_state_dict(torch.load(f"{cwd}/actor.pth", map_location=device))
        
        with torch.no_grad():
            for i in range(env_instance.max_step):
                s_tensor = torch.as_tensor((state,), device=device)
                action = agent.act(s_tensor).detach().cpu().numpy()[0]
                
                # Store the action
                actions_history.append(action)
                
                # Take step in environment
                next_state, reward, done, _, _ = env_instance.step(action)
                
                # Calculate total asset value
                total_asset = env_instance.amount + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
                episode_total_assets.append(total_asset)
                
                # Calculate profit for this step
                profit = total_asset - (episode_total_assets[-2] if len(episode_total_assets) > 1 else initial_asset)
                profits_per_step.append(profit)
                
                # Store portfolio allocation and current prices
                portfolio_allocations.append(env_instance.stocks / total_asset)
                prices_history.append(env_instance.price_ary[env_instance.day])
                
                # Store current date/time (assuming your env tracks this)
                if hasattr(env_instance, 'current_time'):
                    trade_dates.append(env_instance.current_time)
                else:
                    trade_dates.append(i)  # fallback to step number
                
                state = next_state
                if done:
                    break
        
        # Create enhanced visualizations
        
        # 1. Portfolio Value Over Time with dates
        x_axis = trade_dates if isinstance(trade_dates[0], str) else range(len(episode_total_assets))
        fig1 = px.line(
            x=x_axis,
            y=episode_total_assets,
            title='Portfolio Value Over Time',
            labels={'x': 'Time', 'y': 'Total Asset Value ($)'}
        )
        fig1.update_layout(hovermode='x unified')
        fig1.show()
        
        # 2. Profit/Loss per Step with cumulative line
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=list(x_axis) if isinstance(x_axis, range) else x_axis,
            y=profits_per_step,
            name='Step Profit/Loss'
        ))
        cumulative_profits = np.cumsum(profits_per_step)
        fig2.add_trace(go.Scatter(
            x=list(x_axis) if isinstance(x_axis, range) else x_axis,
            y=cumulative_profits,
            name='Cumulative P/L',
            yaxis='y2'
        ))
        fig2.update_layout(
            title='Profit/Loss Analysis',
            yaxis=dict(title='Step P/L ($)'),
            yaxis2=dict(title='Cumulative P/L ($)', overlaying='y', side='right'),
            hovermode='x unified'
        )
        fig2.show()
        
        # 3. Asset Allocation Over Time (enhanced)
        portfolio_allocations = np.array(portfolio_allocations)
        fig3 = go.Figure()
        for i, ticker in enumerate(ticker_list):
            fig3.add_trace(go.Scatter(
                x=list(x_axis) if isinstance(x_axis, range) else x_axis,
                y=portfolio_allocations[:, i],
                name=f'{ticker} Allocation',
                stackgroup='one',
                hovertemplate='%{y:.1%}<extra></extra>'
            ))
        fig3.update_layout(
            title='Portfolio Allocation Over Time',
            xaxis_title='Time',
            yaxis_title='Allocation Ratio',
            hovermode='x unified',
            showlegend=True
        )
        fig3.show()
        
        # 4. Actions Heatmap with enhanced tooltips
        actions_array = np.array(actions_history)
        fig4 = px.imshow(
            actions_array.T,
            labels=dict(x="Time Step", y="Asset", color="Action Value"),
            title="Agent Actions Heatmap",
            y=ticker_list,
            color_continuous_scale='RdYlBu',
            aspect='auto'
        )
        fig4.update_traces(hoverongaps=False)
        fig4.show()
        
        # 5. Asset Prices Over Time
        prices_history = np.array(prices_history)
        fig5 = go.Figure()
        for i, ticker in enumerate(ticker_list):
            fig5.add_trace(go.Scatter(
                x=list(x_axis) if isinstance(x_axis, range) else x_axis,
                y=prices_history[:, i],
                name=f'{ticker} Price',
                hovertemplate='$%{y:,.2f}<extra></extra>'
            ))
        fig5.update_layout(
            title='Asset Prices Over Time',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        fig5.show()
        
        # 6. Enhanced Summary Statistics
        total_profit = episode_total_assets[-1] - initial_asset
        max_drawdown = np.min(episode_total_assets) - initial_asset
        profit_steps = sum(1 for x in profits_per_step if x > 0)
        max_profit_trade = max(profits_per_step)
        max_loss_trade = min(profits_per_step)
        profit_factor = abs(sum(x for x in profits_per_step if x > 0) / sum(x for x in profits_per_step if x < 0)) if sum(x for x in profits_per_step if x < 0) != 0 else float('inf')
        
        print("\n=== Performance Summary ===")
        print(f"Initial Portfolio Value: ${initial_asset:,.2f}")
        print(f"Final Portfolio Value: ${episode_total_assets[-1]:,.2f}")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"Return: {(total_profit/initial_asset)*100:.2f}%")
        print(f"Max Drawdown: ${max_drawdown:,.2f}")
        print(f"Profitable Steps: {profit_steps}/{len(profits_per_step)} ({profit_steps/len(profits_per_step)*100:.2f}%)")
        print(f"Largest Winning Trade: ${max_profit_trade:,.2f}")
        print(f"Largest Losing Trade: ${max_loss_trade:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        return {
            'episode_total_assets': episode_total_assets,
            'profits_per_step': profits_per_step,
            'portfolio_allocations': portfolio_allocations,
            'actions_history': actions_history,
            'prices_history': prices_history,
            'trade_dates': trade_dates,
            'summary_stats': {
                'total_profit': total_profit,
                'return_pct': (total_profit/initial_asset)*100,
                'max_drawdown': max_drawdown,
                'profit_steps': profit_steps,
                'total_steps': len(profits_per_step),
                'max_profit_trade': max_profit_trade,
                'max_loss_trade': max_loss_trade,
                'profit_factor': profit_factor
            }
        }
    
    else:
        raise NotImplementedError("Currently only 'elegantrl' is integrated.")

#### old v1 Test function ############
#######################################

# def test(
#     start_date,
#     end_date,
#     ticker_list,
#     data_source,
#     time_interval,
#     technical_indicator_list,
#     drl_lib,
#     env,
#     model_name,
#     if_vix=False,  # Typically False for crypto
#     **kwargs,
# ):
#     """
#     Test the trained agent on new data (from start_date to end_date),
#     then produce a plot of the total asset value over time.

#     Parameters
#     ----------
#     start_date : str
#         Test data start date (e.g., "2024-01-01")
#     end_date : str
#         Test data end date (e.g., "2024-03-01")
#     ticker_list : list
#         List of asset tickers (e.g., ["BTCUSDT", "ETHUSDT"])
#     data_source : str
#         Name of the data source (e.g., "binance")
#     time_interval : str
#         The resolution of the data (e.g., "5m", "15m", "1D")
#     technical_indicator_list : list
#         List of technical indicators (e.g., ["macd", "rsi", "cci", "dx"])
#     drl_lib : str
#         DRL library name, e.g., "elegantrl"
#     env : class
#         Your environment class, e.g., CryptoTradingEnv
#     model_name : str
#         The model/agent identifier, e.g., "ppo"
#     if_vix : bool, optional
#         Whether to add a volatility index to the data (more for stocks).
#     kwargs : dict
#         Additional arguments, e.g.,
#           - net_dimension : list (your actor/critic network dims)
#           - cwd : str (path to the saved model)
    
#     Returns
#     -------
#     episode_total_assets : list[float]
#         The total asset value at each time step during the test.
#     """
#     print("Received kwargs:", kwargs)
    
#     # 1) Load & process new data with DataProcessor
#     dp = DataProcessor(
#         data_source=data_source,
#         start_date=start_date,
#         end_date=end_date,
#         time_interval=time_interval
#     )
#     # For Crypto, if_vix is typically False, so dp.run(...) won't add VIX/turbulence
#     price_array, tech_array, _ = dp.run(
#         ticker_list=ticker_list,
#         technical_indicator_list=technical_indicator_list,
#         if_vix=if_vix,        # or pass False
#         cache=True
#     )

#     # 2) Build the test environment with if_train=False
#     env_config = {
#         "price_array": price_array,
#         "tech_array": tech_array,
#         "if_train": False,   # This tells your CryptoTradingEnv that it's for inference
#     }
#     env_instance = env(config=env_config)

#     # 3) Load the trained policy
#     net_dimension = kwargs.get("net_dimension", [64, 32])  # Same as training
#     cwd = kwargs.get("cwd", f"./{model_name}")            # Folder with actor.pth
    
#     if drl_lib == "elegantrl":
#         # 4) Get the DRL agent’s predictions on the test data
#         DRLAgent_erl = DRLAgent
#         episode_total_assets = DRLAgent_erl.DRL_prediction(
#             model_name=model_name,
#             cwd=cwd,
#             net_dimension=net_dimension,
#             environment=env_instance,
#         )
#     else:
#         raise NotImplementedError("Currently only 'elegantrl' is integrated.")

#     # 5) Plot the results
#     # -- Option A: Plotly (interactive) --

#     fig = px.line(
#         x=range(len(episode_total_assets)),
#         y=episode_total_assets,
#         labels={'x': 'Time Step', 'y': 'Portfolio Value'},
#         title='Test Performance: Total Asset Value Over Time'
#     )
#     fig.show()

#     # -- Option B: Seaborn (if you prefer static plots) --
#     # import seaborn as sns
#     # import matplotlib.pyplot as plt
#     # sns.set_theme(style="whitegrid")
#     # plt.figure(figsize=(10, 6))
#     # sns.lineplot(x=range(len(episode_total_assets)), y=episode_total_assets)
#     # plt.title("Total Asset Value Over Time (Test)")
#     # plt.xlabel("Time Step")
#     # plt.ylabel("Portfolio Value")
#     # plt.show()

#     return episode_total_assets

        
####### Test functin old ##############
#######################################

# def test(
#     start_date,
#     end_date,
#     ticker_list,
#     data_source,
#     time_interval,
#     technical_indicator_list,
#     drl_lib,
#     env,
#     model_name,
#     if_vix=True,
#     **kwargs,
# ):

#     # import data processor
#     from finrl.meta.data_processor import DataProcessor

#     # fetch data
#     dp = DataProcessor(data_source, **kwargs)
#     data = dp.download_data(ticker_list, start_date, end_date, time_interval)
#     data = dp.clean_data(data)
#     data = dp.add_technical_indicator(data, technical_indicator_list)

#     if if_vix:
#         data = dp.add_vix(data)
#     else:
#         data = dp.add_turbulence(data)
#     price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

#     env_config = {
#         "price_array": price_array,
#         "tech_array": tech_array,
#         "turbulence_array": turbulence_array,
#         "if_train": False,
#     }
#     env_instance = env(config=env_config)

#     # load elegantrl needs state dim, action dim and net dim
#     net_dimension = kwargs.get("net_dimension", 2**7)
#     cwd = kwargs.get("cwd", "./" + str(model_name))
#     print("price_array: ", len(price_array))

#     if drl_lib == "elegantrl":
#         DRLAgent_erl = DRLAgent
#         episode_total_assets = DRLAgent_erl.DRL_prediction(
#             model_name=model_name,
#             cwd=cwd,
#             net_dimension=net_dimension,
#             environment=env_instance,
#         )
#         return episode_total_assets
    
    
    
####################################################
####################################################

#starting the paper trading:

class StockEnvEmpty(gym.Env):
    #Empty Env used for loading rllib agent
    def __init__(self,config):
      state_dim = config['state_dim']
      action_dim = config['action_dim']
      self.env_num = 1
      self.max_step = 10000
      self.env_name = 'StockEnvEmpty'
      self.state_dim = state_dim  
      self.action_dim = action_dim
      self.if_discrete = False  
      self.target_return = 9999
      self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(state_dim,), dtype=np.float32)
      self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        
    def reset(self):
        return 

    def step(self, actions):
        return
############################################
# parameters for training and paper trading.
############################################
data_url = 'wss://data.alpaca.markets'
env = CryptoTradingEnv
ticker_list = CRYPTO_TICKER
action_dim = len(ticker_list) # for training 
print(ticker_list)
print(len(ticker_list))
print(INDICATORS)
# Calculate state dimension components:
# 1 (amount) + 
# action_dim (price_scaled) + 
# action_dim (stocks_scaled) + 
# action_dim (stocks_cool_down) +
# action_dim (position_ratios) +
# len(INDICATORS) * action_dim (technical indicators)
state_dim = 1 + 4 * action_dim + len(INDICATORS) * action_dim
print(f"Calculated state_dim: {state_dim}")
ERL_PARAMS = {
    "learning_rate": 1e-4,
    "batch_size": 4096,
    "gamma": 0.99,
    "seed": 312,
    "net_dimension": [256, 128, 64],  # Adjusted for state dimension
    "target_step": 10000,
    "eval_gap": 50,
    "eval_times": 3
} 
print(f"Creating model with state_dim: {state_dim}, action_dim: {action_dim}")  # Debug print
#############################
# end of the parameters
#############################


# runing the paper trading:

CRYPTO_TICKER_PT = ["BTC/USD"]  # For paper trading
CRYPTO_TICKER_TR = ["BTCUSDT"]  # For training
INDICATORS = ["macd","rsi","cci","dx"]
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
API_BASE_URL = "https://paper-api.alpaca.markets" #new url for crpto wss://stream.data.alpaca.markets/v1beta3/crypto/us
agent_cwd = "/home/souz_wsl/finrl_proj/FinRL_Meta/papertrading_crypto"  # folder containing actor.pth from training
net_dimensions = [128,64]    # same as you used in training
# typical state_dim for:  1 + 3*action_dim + len(INDICATORS)*action_dim
# e.g. 1 + 3*3 + 4*3 = 1 + 9 + 12 = 22 (for 3 tickers, 4 indicators)
# state_dimension = 22
# action_dimension = len(CRYPTO_TICKER)

################ uncomment to run the paper trading ################
##################################################################

# crp_paper_trading = AlpacaPaperTradingCryptoLive(
#         ticker_list=CRYPTO_TICKER_PT,
#         time_interval='5min',
#         drl_lib='elegantrl',
#         agent='ppo',
#         cwd=agent_cwd,
#         net_dim=net_dimensions,
#         state_dim=state_dim,
#         action_dim=action_dim,
#         API_KEY=API_KEY,
#         API_SECRET=API_SECRET,
#         API_BASE_URL=API_BASE_URL,
#         tech_indicator_list=INDICATORS,
       
#         max_stock=12
#     )
# crp_paper_trading.run()



##################################
# start training. function
##################################

# train(start_date='2024-01-01',
#     end_date='2025-01-24',
#     ticker_list=CRYPTO_TICKER_TR, 
#     data_source='binance',
#     time_interval='5m',
#     technical_indicator_list=INDICATORS,
#     drl_lib='elegantrl',
#     env=CryptoTradingEnv,
#     model_name='ppo',
#     if_vix=False,   # for crypto, typically skip 
#     erl_params=ERL_PARAMS,
#     cwd=agent_cwd,
#     break_step=1e6,
#     gpu_id=0,
#     initial_capital=10000  # Set to 10K
# )


##########calling test function################
###############################################
episode_assets = test(
        start_date="2025-01-25",
        end_date="2025-01-27",
        ticker_list=["BTCUSDT"],  # Single asset
        data_source="binance",
        time_interval="5m",
        technical_indicator_list=["macd", "rsi", "cci", "dx"],
        drl_lib="elegantrl",
        env=CryptoTradingEnv,
        model_name="ppo",
        if_vix=False,
        net_dimension=[256, 128, 64],  # Updated net dimensions
        cwd=agent_cwd,   # folder that has 'actor.pth'
        initial_capital=10000  # Set to 10K
    )
