import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from core.model import BaseMuZeroNet, renormalize


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        init_zero=False,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    # TODO: [future] add residual in mlp if given more than 2 hidden layers
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2,
                                              downsample=self.conv2)
        self.resblocks2 = nn.ModuleList(
            # [ResidualBlock(out_channels) for _ in range(3)]
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            # [ResidualBlock(out_channels) for _ in range(3)]
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # TODO: [finished] add bn + relu
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        # TODO: [finished] use downsample block in place of conv2
        # x = self.conv2(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class RepresentationNetwork(nn.Module):
    def __init__(
            self,
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            downsample,
            momentum=0.1,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels - 1,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels - 1,
        )
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels-1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

    def forward(self, x, spd, debug):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        if debug:
            ipdb.set_trace()

        spd = torch.tensor(spd).to(x.device).float()
        spd = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(spd.device) * spd[:, None, None, None]
        x = torch.cat((x, spd), 1)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


class DynamicsNetwork(nn.Module):
    def __init__(
            self,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            fc_reward_layers,
            full_support_size,
            block_output_size_reward,
            lstm_hidden_size=64,
            momentum=0.1,
            init_zero=False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size

        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.reward_resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                      momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:, :-1, :, :]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        # TODO: [finished] add some res blocks here (followed by prediction part)
        # for block in self.reward_resblocks:
        #     x = block(x)

        x = self.conv1x1_reward(x)
        # TODO: [finished] add bn + relu
        x = self.bn_reward(x)
        x = nn.functional.relu(x)

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        # reward = self.fc(x)
        reward_sum, reward_hidden = self.lstm(x, reward_hidden)
        reward_sum = reward_sum.squeeze(0)
        # TODO: use output or c or h or concat all as input of fc
        reward_sum = self.bn_reward_sum(reward_sum)
        reward_sum = nn.functional.relu(reward_sum)
        reward_sum = self.fc(reward_sum)

        return state, reward_hidden, reward_sum

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)
        # reward_mean = np.abs(temp_weights).tolist()

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# TODO: [future] change prediction fc to model similar to resnet (resblock+pooling+fc)
class PredictionNetwork(nn.Module):
    def __init__(
            self,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy,
            observation_shape,
            momentum=0.1,
            init_zero=False,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )
        self.reduced_channels_value = reduced_channels_value
        self.reduced_channels_policy = reduced_channels_policy
        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero,
                            momentum=momentum)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size, init_zero=init_zero,
                             momentum=momentum)

        policy_branch_list, value_branch_list = [], []
        for i in range(6):
            policy_branch_list.append(nn.Sequential(
                self.conv1x1_policy,
                self.bn_policy,
                nn.ReLU(True)
            ))
            value_branch_list.append(nn.Sequential(
                self.conv1x1_value,
                self.bn_value,
                nn.ReLU(True)
            ))
        self.policy_branches = nn.ModuleList(policy_branch_list)
        self.value_branches = nn.ModuleList(value_branch_list)
        self.obs_shape = observation_shape

    def forward(self, x, cmd, start_training):
        for block in self.resblocks:
            x = block(x)
        # value = self.conv1x1_value(x)
        # # TODO: [finished] add bn + relu
        # value = self.bn_value(value)
        # value = nn.functional.relu(value)
        #
        # policy = self.conv1x1_policy(x)
        # # TODO: [finished] add bn + relu
        # policy = self.bn_policy(policy)
        # policy = nn.functional.relu(policy)

        if start_training is True:
            ipdb.set_trace()

        cmd = torch.tensor(cmd).to(x.device).float()
        hidden_h, hidden_w = int(self.obs_shape[1] / 16), int(self.obs_shape[2] / 16)

        value = torch.zeros((x.shape[0], self.reduced_channels_value, hidden_h, hidden_w)).to(x.device)
        for i, branch in enumerate(self.policy_branches):
            control = (i == cmd).expand(self.reduced_channels_policy, hidden_h, hidden_w, -1).permute(3, 0, 1, 2)
            value += branch(x) * control

        policy = torch.zeros((x.shape[0], self.reduced_channels_policy, hidden_h, hidden_w)).to(x.device)
        for i, branch in enumerate(self.value_branches):
            control = (i == cmd).expand(self.reduced_channels_value, hidden_h, hidden_w, -1).permute(3, 0, 1, 2)
            policy += branch(x) * control

        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroNet(BaseMuZeroNet):
    def __init__(
            self,
            observation_shape,
            stacked_observations,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            reduced_channels_value,
            reduced_channels_policy,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            reward_support_size,
            value_support_size,
            downsample,
            inverse_value_transform,
            inverse_reward_transform,
            lstm_hidden_size,
            bn_mt=0.1,
            proj_hid=256,
            proj_out=256,
            pred_hid=64,
            pred_out=256,
            init_zero=False,
            state_norm=False
    ):
        super(MuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size
        block_output_size_reward = (
            (
                    reduced_channels_reward
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                    reduced_channels_value
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                    reduced_channels_policy
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = RepresentationNetwork(
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            downsample,
            momentum=bn_mt,
        )

        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels + 1,
            reduced_channels_reward,
            fc_reward_layers,
            reward_support_size,
            block_output_size_reward,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            block_output_size_value,
            block_output_size_policy,
            observation_shape,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        in_dim = num_channels * math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
        self.porjection_in_dim = in_dim
        self.projection = nn.Sequential(
            nn.Linear(self.porjection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

        self.d = D()

    def prediction(self, encoded_state, cmd, start_training=False):
        policy, value = self.prediction_network(encoded_state, cmd, start_training)
        return policy, value

    def representation(self, observation, spd, debug=False):
        encoded_state = self.representation_network(observation, spd, debug)
        if not self.state_norm:
            return encoded_state
        else:
            # # Scale encoded state between [0, 1] (See appendix paper Training)
            # min_encoded_state = (
            #     encoded_state.view(
            #         -1,
            #         encoded_state.shape[1],
            #         encoded_state.shape[2] * encoded_state.shape[3],
            #     )
            #     .min(2, keepdim=True)[0]
            #     .unsqueeze(-1)
            # )
            # max_encoded_state = (
            #     encoded_state.view(
            #         -1,
            #         encoded_state.shape[1],
            #         encoded_state.shape[2] * encoded_state.shape[3],
            #     )
            #     .max(2, keepdim=True)[0]
            #     .unsqueeze(-1)
            # )
            # scale_encoded_state = max_encoded_state - min_encoded_state
            # scale_encoded_state[scale_encoded_state < 1e-5] = 1e-5
            # encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
            # return encoded_state_normalized
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
                .to(action.device)
                .float()
        )
        action_one_hot = (
                action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, reward_sum = self.dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, reward_sum
        else:
            # Scale encoded state between [0, 1] (See paper appendix Training)
            # min_next_encoded_state = (
            #     next_encoded_state.view(
            #         -1,
            #         next_encoded_state.shape[1],
            #         next_encoded_state.shape[2] * next_encoded_state.shape[3],
            #     )
            #     .min(2, keepdim=True)[0]
            #     .unsqueeze(-1)
            # )
            # max_next_encoded_state = (
            #     next_encoded_state.view(
            #         -1,
            #         next_encoded_state.shape[1],
            #         next_encoded_state.shape[2] * next_encoded_state.shape[3],
            #     )
            #     .max(2, keepdim=True)[0]
            #     .unsqueeze(-1)
            # )
            # scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
            # scale_next_encoded_state[scale_next_encoded_state < 1e-5] = 1e-5
            # next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, reward_sum

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def contrastive_loss(self, x1, x2):
        out1 = x1.view(-1, self.porjection_in_dim)

        z1 = self.projection(out1)
        p1 = self.projection_head(z1)

        out2 = x2.view(-1, self.porjection_in_dim)
        z2 = self.projection(out2)
        p2 = self.projection_head(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.

        return d1 + d2

    def project(self, hidden_state, with_grad=True):
        hidden_state = hidden_state.view(-1, self.porjection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, p, z):
        z = z.detach()

        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1)
