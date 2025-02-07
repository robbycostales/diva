# Modified from https://github.com/lmzintgraf/varibad/tree/master
import torch
import torch.nn as nn
from torch.nn import functional as F

from diva.environments import utils as utl


class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 pred_type='deterministic',
                 state_feature_extractor=utl.FeatureExtractor
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.args = args

        self.state_encoder = state_feature_extractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions):

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        ha = self.action_encoder(actions)
        hs = self.state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 state_feature_extractor=utl.FeatureExtractor
                 ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if self.multi_head:
            # one output head per state to predict rewards
            curr_input_dim = latent_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]
            self.fc_out = nn.Linear(curr_input_dim, num_states)
        else:
            # get state as input and predict reward prob
            self.state_encoder = state_feature_extractor(
                state_dim, state_embed_dim, F.relu)
            if self.input_action:
                self.action_encoder = utl.FeatureExtractor(
                    action_dim, action_embed_dim, F.relu)
            else:
                self.action_encoder = None
            curr_input_dim = latent_dim + state_embed_dim
            if input_prev_state:
                curr_input_dim += state_embed_dim
            if input_action:
                curr_input_dim += action_embed_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]

            if pred_type == 'gaussian':
                self.fc_out = nn.Linear(curr_input_dim, 2)
            else:
                self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):
        # we do the action-normalisation (the the env bounds) here
        if actions is not None:
            actions = utl.squash_action(actions, self.args)

        if self.multi_head:
            h = latent_state.clone()
        else:
            hns = self.state_encoder(next_state)
            # TODO: This is a hacky way to deal with single-process case where
            #       hns.shape is e.g. [10, 400, 1, 8] and latent_state.shape is
            #       e.g. [10, 400, 5]. To correct, for now, we are adding a
            #       dimension to latent_state.shape to make it e.g. 
            #       [10, 400, 1, 5]
            if len(hns.shape) == 4 and len(latent_state.shape) == 3:
                print('WARNING: shape mismatch in RewardDecoder.forward()')
                latent_state = latent_state.unsqueeze(2)

            h = torch.cat((latent_state, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                 ):
        super(TaskDecoder, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        print('pred_type:', pred_type)
        print('task_dim:', task_dim)
        print('output_dim:', output_dim)
        print('curr_input_dim:', curr_input_dim)
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):

        h = latent_state

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)
