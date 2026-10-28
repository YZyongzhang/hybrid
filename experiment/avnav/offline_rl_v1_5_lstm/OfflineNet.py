import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC_LSTM_CQL_v1_5(nn.Module):
    """
    Offline SAC-CQL with LSTM temporal encoder.
    Input states are pre-encoded features from VADE (shape: [B, T, D]).
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        beta,
        device,
        lstm_hidden_dim=256,
        lstm_num_layers=1,
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.target_entropy = target_entropy
        self.clip_grad_param = 1.0

        self.temporal_encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        ).to(device)

        self.actor = PolicyNet(lstm_hidden_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(lstm_hidden_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(lstm_hidden_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(lstm_hidden_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(lstm_hidden_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.temporal_optimizer = torch.optim.Adam(self.temporal_encoder.parameters(), lr=actor_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(1.0), dtype=torch.float32, device=device)
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def _to_seq_batch(self, seq):
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif seq.dim() == 2:
            seq = seq.unsqueeze(0)
        return seq.float().to(self.device)

    def _encode_sequence(self, seq):
        seq = self._to_seq_batch(seq)
        encoded, _ = self.temporal_encoder(seq)
        return encoded

    def _flatten_time(self, x):
        return x.reshape(-1, x.shape[-1])

    def _reshape_vector(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 0:
            return x.view(1)
        return x.reshape(-1)

    def get_action(self, state_seq, eval=False):
        with torch.no_grad():
            encoded = self._encode_sequence(state_seq)
            latest = encoded[:, -1, :]
            probs = self.actor(latest)
            if eval:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.distributions.Categorical(probs).sample()
        return int(action.item()) if action.numel() == 1 else action

    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
            q1_value = self.target_critic_1(next_states)
            q2_value = self.target_critic_2(next_states)
            min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
            next_value = min_qvalue + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value.squeeze(1) * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, states, actions, rewards, next_states, dones):
        seq_states = self._encode_sequence(states)
        seq_next_states = self._encode_sequence(next_states)

        states_flat = self._flatten_time(seq_states)
        next_states_flat = self._flatten_time(seq_next_states)
        # states_actor = states_flat.detach()

        rewards = self._reshape_vector(rewards).float().to(self.device)
        dones = self._reshape_vector(dones).float().to(self.device)
        actions = self._reshape_vector(actions).long().to(self.device).unsqueeze(1)

        probs = self.actor(states_flat)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states_flat).detach()
        q2_value = self.critic_2(states_flat).detach()
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        # 更新 critic 网络，重新进行LSTM的encode
        seq_states = self._encode_sequence(states)
        seq_next_states = self._encode_sequence(next_states)

        states_flat = self._flatten_time(seq_states)
        next_states_flat = self._flatten_time(seq_next_states)
        td_target = self.calc_target(rewards, next_states_flat, dones).detach()

        critic_1_q_values = self.critic_1(states_flat)
        critic_1_q_taken = critic_1_q_values.gather(1, actions).squeeze(1)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_taken, td_target))

        critic_2_q_values = self.critic_2(states_flat)
        critic_2_q_taken = critic_2_q_values.gather(1, actions).squeeze(1)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_taken, td_target))
        
        # 从公式上来看，cql这里的实现有一些问题。目前先不动这个。控制变量。
        # cql1_scaled_loss = torch.logsumexp(critic_1_q_values, dim=1).mean() - critic_1_q_values.mean()
        # cql2_scaled_loss = torch.logsumexp(critic_2_q_values, dim=1).mean() - critic_2_q_values.mean()
        cql1_scaled_loss = torch.logsumexp(critic_1_q_values, dim=1).mean() - critic_1_q_taken.mean()
        cql2_scaled_loss = torch.logsumexp(critic_2_q_values, dim=1).mean() - critic_2_q_taken.mean()

        cql_1_loss = critic_1_loss + self.beta * cql1_scaled_loss
        cql_2_loss = critic_2_loss + self.beta * cql2_scaled_loss
        total_critic_loss = cql_1_loss + cql_2_loss

        self.temporal_optimizer.zero_grad()
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        total_critic_loss.backward()

        clip_grad_norm_(self.temporal_encoder.parameters(), self.clip_grad_param)
        clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_param)
        clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_param)

        self.temporal_optimizer.step()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return {
            "critic1_loss": critic_1_loss.item(),
            "critic2_loss": critic_2_loss.item(),
            "cql1_scaled_loss": cql1_scaled_loss.item(),
            "cql2_scaled_loss": cql2_scaled_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "cql_1_loss": cql_1_loss.item(),
            "cql_2_loss": cql_2_loss.item(),
            "entropy": entropy.mean().item(),
            "alpha": self.log_alpha.exp().item(),
        }