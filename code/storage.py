import torch

class RolloutStorage():
    def __init__(self, num_steps, num_processes):
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)

        self.advantages = torch.zeros(num_steps + 1, num_processes, 1)
        self.tdlamret = torch.zeros(num_steps + 1, num_processes, 1)
        self.lastgaelam = torch.zeros(num_processes, 1)

    def to(self, device):
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.masks = self.masks.to(device)
        self.advantages = self.advantages.to(device)
        self.tdlamret = self.tdlamret.to(device)
        self.lastgaelam = self.lastgaelam.to(device)

    def insert(self, step, reward, mask, value):
        self.masks[step + 1].copy_(mask)
        self.rewards[step + 1].copy_(reward)
        self.value_preds[step + 1].copy_(value)

    def after_update(self):
        self.masks[0].copy_(self.masks[-1])
        self.rewards[0].copy_(self.rewards[-1])
        self.value_preds[0].copy_(self.value_preds[-1])

    def compute_vtarg_and_adv(self, next_value, gamma, lam):

        self.lastgaelam.zero_()
        for step in reversed(range(self.rewards.size(0) - 1)):
            nonterminal = self.masks[step + 1]
            delta = self.rewards[step + 1] + gamma * next_value * nonterminal - self.value_preds[step + 1]
            self.advantages[step + 1] = self.lastgaelam = delta + gamma * lam * nonterminal * self.lastgaelam
            next_value = self.value_preds[step + 1]
        self.tdlamret = self.advantages + self.value_preds
