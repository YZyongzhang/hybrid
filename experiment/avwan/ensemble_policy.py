"""Average probabilities from complementary pretrained acoustic adaptations."""
import numpy as np
import torch

from .model.hybrid import Network


class FoundationEnsemblePolicy:
    def __init__(self, state, device):
        self.device = device
        self.models = []
        for member in state['members']:
            model = Network(action_count=state['signature']['actions'], **member['network']).to(device)
            model.load_state_dict(member['model'], strict=True)
            self.models.append(model.eval().requires_grad_(False))
        self.weights = torch.tensor(state['weights'], device=device, dtype=torch.float32)
        if (not self.models or len(self.models) != len(self.weights)
                or not torch.isfinite(self.weights).all() or (self.weights <= 0).any()):
            raise ValueError('Ensemble members need positive matching weights')
        self.weights /= self.weights.sum()
        self.reset()

    def reset(self):
        self.previous = None
        if getattr(self, 'action_selector', None):self.action_selector.reset()

    @torch.inference_mode()
    def act(self, obs, action_mask):
        previous = self.previous
        rgb = np.concatenate((previous['rgb'] if previous else np.zeros_like(obs['rgb']), obs['rgb']), -1)
        depth = np.concatenate((previous['depth'] if previous else np.zeros_like(obs['depth']), obs['depth']), -1)
        values = [obs['spectrogram'], rgb.astype(np.float32)/255., depth, obs['gm'], obs['am'], action_mask]
        tensors = [torch.as_tensor(np.ascontiguousarray(x),device=self.device).unsqueeze(0) for x in values]
        probabilities = torch.stack([m.forward_joint(*tensors)[0].softmax(-1)[0] for m in self.models])
        mixture = (probabilities*self.weights[:,None]).sum(0)
        mixture = mixture.masked_fill(~tensors[-1][0].bool(),0.)
        selector = getattr(self,'action_selector',None)
        action = selector(obs,action_mask,mixture.clamp_min(1e-30).log()) if selector else int(mixture.argmax())
        self.previous = {k:np.asarray(obs[k]).copy() for k in ['rgb','depth']}
        return action
