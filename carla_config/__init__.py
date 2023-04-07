import gym
import ipdb
import torch
from gym.wrappers import TimeLimit
# from baselines.common.atari_wrappers import WarpFrame, EpisodicLifeEnv
from core.config import BaseMuZeroConfig, DiscreteSupport
from core.utils import make_carla
from .env_wrapper import CarlaWrapper
from core.dataset import Transforms
from .model import MuZeroNet
import carla_gym


class CarlaConfig(BaseMuZeroConfig):
    def __init__(self):
        super(CarlaConfig, self).__init__(
            training_steps=1500000,
            last_steps=0,
            test_interval=10000,
            log_interval=100,
            vis_interval=1000,
            test_episodes=1,
            checkpoint_interval=300,
            target_model_interval=200,
            save_ckpt_interval=10000,
            max_moves=5000,
            test_max_moves=20000,
            history_length=50001,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.006,
            num_simulations=50,
            batch_size=64,# 256,
            td_steps=5,
            num_actors=2,  # number of DataWorkers
            # network initialization/ & normalization
            off_correction=True,
            gray_scale=False,
            change_temperature=False,
            episode_life=True,
            init_zero=True,
            state_norm=False,
            clip_reward=False,
            random_start=True,
            # storage efficient
            cvt_string=True,
            image_based=True,
            # lr scheduler
            lr_warm_up=0.01,
            lr_type='step',
            lr_init=0.2,
            lr_decay_rate=0.1,
            # ToDo: change
            lr_decay_steps=500000,
            # replay window
            auto_td_steps=30 * 1000,
            start_window_size=10,
            window_size=125000,
            total_transitions=50 * 1000 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=4,
            three_cam=True,
            # spr
            consist_type='spr',
            consistency_coeff=2,
            # coefficient
            uniform_ratio=0,
            priority_reward_ratio=0,
            tail_ratio=0,
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            # reward sum
            lstm_horizon_len=1,
            lstm_hidden_size=64,
            # value reward support
            value_support=DiscreteSupport(-300, 300, delta=1),
            reward_support=DiscreteSupport(-300, 300, delta=1))

        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_window_size = self.start_window_size * 100 // self.frame_skip
        self.start_window_size = max(1, int(self.start_window_size))

        if not self.off_correction:
            self.auto_td_steps = self.training_steps

        self.bn_mt = 0.1

        # TODO: use smaller networks to speed up, 4 blocks
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * self.training_steps:
                return 1.0
            elif trained_steps < 0.75 * self.training_steps:
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name

        # gray scale
        if self.gray_scale:
            self.image_channel = 1

        if self.three_cam:
            obs_shape = (self.image_channel, 96, 192)
        else:
            obs_shape = (self.image_channel, 96, 96)

        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])
        self.action_space_size = 28

    def get_uniform_network(self):
        return MuZeroNet(
            self.obs_shape,
            self.stacked_observations,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, test=False, final_test=False, **kwargs):
        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip
            else:
                max_moves = self.test_max_moves
            env = make_carla(self.env_name, max_moves, **kwargs)
        else:
            env = make_carla(self.env_name, self.max_moves, **kwargs)

        return CarlaWrapper(env, discount=self.discount, cvt_string=self.cvt_string, test=test)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


muzero_config = CarlaConfig()
