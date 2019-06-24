"""
BMIL model class
    - Defines the various neural networks - policy, discriminator, and {forward, backward, action} predictions
    - forward() creates computational graph with on-policy data
    - forward_offPol() creates computational graph with off-policy data
"""

import numpy as np
import namedlist
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy import DiagGaussian
import encoder_decoder
from utils.helpers import l2_loss_criterion, RunningMeanStd, activation

# Container to return all required values from model
ModelReturn = namedlist.namedlist('ModelReturn', [
    ('belief_state', None),
    ('discriminator_out_d', None),
    ('discriminator_out_b', None),
    ('value_estimate', None),
    ('action', None),
    ('ac_log_probs', None),
    ('dist_entropy', None),
    ('reg_loss', None),
    ('reg_loss_item', None)
])

class BMIL(nn.Module):
    def __init__(self,
                 device,
                 ac_space,
                 obs_dim,
                 action_encoding_dim,
                 discr_action_encoding_dim,
                 obs_encoder_hidden,
                 belief_dim,
                 belief_loss_type,
                 belief_regularization,
                 init_function,
                 detach_belief_module,
                 forward_jump,
                 backward_jump,
                 num_processes,
                 num_expert_processes,
                 init_logstd,
                 nonlinearity,
                 **kwargs
                 ):
        super().__init__()
        self.device = device
        self.belief_dim = belief_dim
        self.belief_loss_type = belief_loss_type
        self.belief_regularization = belief_regularization
        self.init_function = init_function
        self.agent_bs = num_processes - num_expert_processes
        self.detach_belief_module = detach_belief_module
        self.fwd_jump = forward_jump
        self.bwd_jump = backward_jump

        nonlinearity, gain = activation(nonlinearity)

        # For observation normalization
        self.ob_rms = RunningMeanStd(shape=obs_dim)

        obs_encoding_dim = obs_encoder_hidden[-1]
        ac_shape = ac_space.shape[0]

        # Belief RNN
        self.belief_gru = nn.GRUCell(obs_encoding_dim + action_encoding_dim, belief_dim)

        # Observation Encoder.
        self.obs_encoder = encoder_decoder.get_encoder(
            obs_dim,
            obs_encoder_hidden,
            nonlinearity
        )

        # Action Encoder.
        self.action_encoder = nn.Sequential(
            nn.Linear(ac_shape, action_encoding_dim),
            nonlinearity())

        # Decoder for 1-step forward prediction (b_{t-1}, a_{t-1} -> o_t)
        self.decoder_fwd_1step = encoder_decoder.get_decoder(
            input_dim=belief_dim + action_encoding_dim,
            output_dim=obs_dim,
            decoder_hidden=obs_encoder_hidden,
            nonlinearity=nonlinearity)

        # Decoder for 1-step backward prediction (b_t, a_{t-1} -> o_{t-1})
        self.decoder_bwd_1step = encoder_decoder.get_decoder(
            input_dim=belief_dim + action_encoding_dim,
            output_dim=obs_dim,
            decoder_hidden=obs_encoder_hidden,
            nonlinearity=nonlinearity)

        # Decoder for k-step forward prediction (b_{t-1}, a_{t-1}:a_{t+k-1} -> o_{t+k})
        self.decoder_fwd_kstep = encoder_decoder.get_decoder(
            input_dim=belief_dim + action_encoding_dim,
            output_dim=obs_dim,
            decoder_hidden=obs_encoder_hidden,
            nonlinearity=nonlinearity)

        # Decoder for k-step backward prediction (b_{t}, a_{t-1}:a_{t-k-1} -> o_{t-k-1})
        self.decoder_bwd_kstep = encoder_decoder.get_decoder(
            input_dim=belief_dim + action_encoding_dim,
            output_dim=obs_dim,
            decoder_hidden=obs_encoder_hidden,
            nonlinearity=nonlinearity)

        # Decoder for 1-step action prediction (b_{t-1}, o_t -> a_{t-1})
        self.decoder_ac_1step = encoder_decoder.get_decoder(
            input_dim=belief_dim + obs_dim,
            output_dim=ac_shape,
            decoder_hidden=obs_encoder_hidden,
            nonlinearity=nonlinearity)

        # Decoder for k-step action prediction (b_{t-1}, o_{t+k} -> a_{t-1}:a_{t+k-1})
        self.decoder_ac_kstep = encoder_decoder.get_decoder(
            input_dim=belief_dim + obs_dim,
            output_dim=ac_shape*(self.fwd_jump+1),
            decoder_hidden=obs_encoder_hidden,
            nonlinearity=nonlinearity)

        # Convolution layers to encode (future/past) action sequences
        kernel_size = 3
        stride = padding = 1
        num_filters1 = num_filters2 = 5

        self.conv1_future_seq = nn.Conv1d(self.fwd_jump + 1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_future_seq = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convLinear_future = nn.Linear(num_filters2*ac_shape, action_encoding_dim)

        self.conv1_past_seq = nn.Conv1d(self.bwd_jump + 1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_past_seq = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convLinear_past = nn.Linear(num_filters2*ac_shape, action_encoding_dim)

        # Discriminator action encoder.
        self.discriminator_ac_encoder = nn.Sequential(
                nn.Linear(ac_shape, discr_action_encoding_dim),
                nonlinearity())

        # Discriminator classifier D(b_t, o_t, a_t), output: probability
        self.discriminator_network = nn.Sequential(
                nn.Linear(belief_dim + obs_dim + discr_action_encoding_dim, 64),
                nonlinearity(),
                nn.Linear(64, 64),
                nonlinearity(),
                nn.Linear(64, 1),
                nn.Sigmoid()
        )

        # ===========
        # Actor-Critic
        # ===========

        actor_hidden_dim = critic_hidden_dim = 64

        # Value function, V(b_t, o_t)
        self.critic = nn.Sequential(
                    nn.Linear(belief_dim + obs_dim, critic_hidden_dim),
                    nonlinearity(),
                    nn.Linear(critic_hidden_dim, critic_hidden_dim),
                    nonlinearity(),
                    nn.Linear(critic_hidden_dim, 1))

        # Policy network, pi(a_t|b_t,o_t)
        self.actor = nn.Sequential(
                    nn.Linear(belief_dim + obs_dim, actor_hidden_dim),
                    nonlinearity(),
                    nn.Linear(actor_hidden_dim, actor_hidden_dim),
                    nonlinearity())

        assert ac_space.__class__.__name__ == "Box", "Current implementation only supports continuous action spaces."
        self.action_dist = DiagGaussian(actor_hidden_dim, ac_shape, init_logstd)

        self.train()
        self.reset_parameters(gain)

    def reset_parameters(self, gain):
        def weights_init(gain):
            def fn(m):
                classname = m.__class__.__name__
                init_func = getattr(torch.nn.init, self.init_function)
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    init_func(m.weight.data, gain=gain)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                if classname.find('GRUCell') != -1:
                    init_func(m.weight_ih.data)
                    init_func(m.weight_hh.data)
                    m.bias_ih.data.fill_(0)
                    m.bias_hh.data.fill_(0)

            return fn

        self.apply(weights_init(gain))
        if self.action_dist.__class__.__name__ == "DiagGaussian":
            self.action_dist.fc_mean.weight.data.mul_(0.01)

    def _normalize(self, tnsr, update_rms):
        """
        Normalize (and clip) observations tensors
        """

        if update_rms:
            self.ob_rms.update(tnsr.numpy())

        tnsr_np = np.clip((tnsr.numpy() - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8), -10., 10.)
        return torch.from_numpy(tnsr_np).float().to(self.device)

    def _encode(self, *, ob, prev_ob, ob_tpk, future_k_acs,
            future_mask, ob_tmkm1, past_k_acs, past_mask,
            prev_ac, prev_belief_state):
        """
        Encode ob, prev_ac and prev_belief_state into the new belief_state with GRU,
        and compute all belief regularization losses
        """

        encoded_ac = self.action_encoder(prev_ac)
        x = self.obs_encoder(ob)
        x = torch.cat([x, encoded_ac], dim=1)

        # Update belief
        belief_state = self.belief_gru(x, prev_belief_state)

        # Prediction losses
        fwd_1step_loss = bwd_1step_loss = ac_1step_loss = \
                fwd_kstep_loss = bwd_kstep_loss = \
                ac_kstep_loss = torch.zeros(x.size(0)).to(self.device)

        # Task-agnostic belief learning uses 1-step forward prediction
        if self.belief_loss_type == 'task_agnostic' or self.belief_regularization:

            # 1-step forward prediction.
            obs_predicted = self.decoder_fwd_1step(
                torch.cat([
                    prev_belief_state,
                    encoded_ac
                ], dim=1))
            fwd_1step_loss = l2_loss_criterion(obs_predicted, ob)

        # In addition to 1-step forward prediction, the belief is regularized with
        # k-step forward, {1,k}-step backward & {1,k}-step action predictions
        if self.belief_regularization:

            # 1-step backward prediction.
            obs_predicted = self.decoder_bwd_1step(
                torch.cat([
                    belief_state,
                    encoded_ac
                ], dim=1))
            bwd_1step_loss = l2_loss_criterion(obs_predicted, prev_ob)

            # 1-step action prediction.
            predicted_ac = self.decoder_ac_1step(
                    torch.cat([
                        prev_belief_state,
                        ob
                    ], dim=1))
            ac_1step_loss = l2_loss_criterion(predicted_ac, prev_ac)

            # k-step forward prediction.
            if ob_tpk is not None:
                future_acs = torch.cat([prev_ac, future_k_acs], dim=1)
                future_acs = torch.stack(torch.chunk(future_acs, self.fwd_jump + 1, dim=1)).transpose(1, 0)

                x = F.relu(self.conv1_future_seq(future_acs))
                x = F.relu(self.conv2_future_seq(x))
                encoded_future_acs = self.convLinear_future(x.view(x.size(0), -1))

                obs_predicted = self.decoder_fwd_kstep(
                    torch.cat([
                        prev_belief_state,
                        encoded_future_acs
                    ], dim=1))
                fwd_kstep_loss = l2_loss_criterion(obs_predicted * future_mask, ob_tpk * future_mask)

            # k-step backward prediction.
            if ob_tmkm1 is not None:
                past_acs = torch.cat([prev_ac, past_k_acs], dim=1)
                past_acs = torch.stack(torch.chunk(past_acs, self.bwd_jump + 1, dim=1)).transpose(1, 0)

                x = F.relu(self.conv1_past_seq(past_acs))
                x = F.relu(self.conv2_past_seq(x))
                encoded_past_acs = self.convLinear_past(x.view(x.size(0), -1))

                obs_predicted = self.decoder_bwd_kstep(
                    torch.cat([
                        belief_state,
                        encoded_past_acs
                    ], dim=1))
                bwd_kstep_loss = l2_loss_criterion(obs_predicted * past_mask, ob_tmkm1 * past_mask)

            # k-step action prediction.
            if ob_tpk is not None:
                predicted_acs = self.decoder_ac_kstep(
                        torch.cat([
                            prev_belief_state,
                            ob_tpk
                        ], dim=1))
                future_acs = torch.cat([prev_ac, future_k_acs], dim=1).detach()
                ac_kstep_loss = l2_loss_criterion(predicted_acs * future_mask, future_acs * future_mask)

        # loss accumulation
        reg_loss = (
                fwd_1step_loss
                + fwd_kstep_loss
                + bwd_1step_loss
                + bwd_kstep_loss
                + ac_1step_loss
                + ac_kstep_loss)

        return belief_state, reg_loss, torch.mean(reg_loss).item()

    def forward(self, curr_memory):
        """
        Forward pass to compute all losses with on-policy data
        """

        normalized_ob = self._normalize(curr_memory['curr_ob'], update_rms=True)
        normalized_prev_ob = self._normalize(curr_memory['prev_ob'], update_rms=False)

        belief_state, reg_loss, reg_loss_item = self._encode(
                ob=normalized_ob,
                prev_ob=normalized_prev_ob,
                prev_ac=curr_memory['prev_ac'].detach(),
                prev_belief_state=curr_memory['prev_belief'],
                ob_tpk=None, future_k_acs=None,    # Currently, k-step losses are computed only with off-policy data in forward_offPol()
                future_mask=None, ob_tmkm1=None,
                past_k_acs=None, past_mask=None)

        # Detach belief-state for Task-Agnostic belief learning
        encoded_state = belief_state.detach() if self.detach_belief_module else belief_state

        model_return = ModelReturn()
        model_return.belief_state = belief_state
        model_return.reg_loss = reg_loss
        model_return.reg_loss_item = reg_loss_item

        # Run actor & critic
        model_return.value_estimate = self.critic(torch.cat([encoded_state, normalized_ob], dim=1))
        actor_state = self.actor(torch.cat([encoded_state, normalized_ob], dim=1))
        action = self.action_dist.sample(actor_state, deterministic=False)

        # first #agent_bs actions are sampled from the actor, rest are pre-defined expert actions (from database)
        model_return.action = torch.cat([action[:self.agent_bs],
            curr_memory['expert_ac'][self.agent_bs:].type(action.type())], dim=0)

        discriminator_acs = self.discriminator_ac_encoder(model_return.action.detach())
        discriminator_in = torch.cat([encoded_state.detach(), normalized_ob, discriminator_acs], dim=1)
        model_return.discriminator_out_d = self.discriminator_network(discriminator_in)

        # discriminator_out_d stores the discriminator output for computing the gradient for discriminator, whereas
        # discriminator_out_b stores the discriminator output for computing the gradient for the belief RNN
        discriminator_in = torch.cat([encoded_state, normalized_ob, discriminator_acs.detach()], dim=1)
        model_return.discriminator_out_b = self.discriminator_network(discriminator_in)

        model_return.ac_log_probs, model_return.dist_entropy = self.action_dist.logprobs_and_entropy(actor_state, action.detach())
        return model_return

    def forward_offPol(self, curr_memory_offPol):
        """
        Forward pass to compute regularization losses with off-policy data
        """

        normalized_ob, normalized_prev_ob, normalized_ob_tpk, normalized_ob_tmkm1 = list(map(lambda x: self._normalize(x, update_rms=False),
            [curr_memory_offPol['curr_ob'], curr_memory_offPol['prev_ob'], curr_memory_offPol['ob_tpk'], curr_memory_offPol['ob_tmkm1']]))

        belief_state, reg_loss, reg_loss_item = self._encode(
                ob=normalized_ob,
                prev_ob=normalized_prev_ob,
                prev_ac=curr_memory_offPol['prev_ac'].to(self.device),
                prev_belief_state=curr_memory_offPol['prev_belief'],
                ob_tpk=normalized_ob_tpk,
                future_k_acs=curr_memory_offPol['future_k_acs'].to(self.device),
                future_mask=curr_memory_offPol['future_mask'].to(self.device),
                ob_tmkm1=normalized_ob_tmkm1,
                past_k_acs=curr_memory_offPol['past_k_acs'].to(self.device),
                past_mask=curr_memory_offPol['past_mask'].to(self.device))

        model_return = ModelReturn()
        model_return.belief_state = belief_state
        model_return.reg_loss = reg_loss
        model_return.reg_loss_item = reg_loss_item
        return model_return
