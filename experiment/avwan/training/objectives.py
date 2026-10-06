"""Joint supervision through the user-supplied hybrid.Network interface."""
import torch
from torch.nn import functional as F
from ..model.maps import unpack_geometry


def objective(model, batch, config):
    rgb = torch.cat((batch['prev_rgb'], batch['rgb']), dim=-1).float() / 255.
    depth = torch.cat((batch['prev_depth'], batch['depth']), dim=-1).float()
    action_logits, angle_logits = model.forward_joint(
        batch['spectrogram'], rgb, depth, unpack_geometry(batch['gm']), batch['am'], batch['action_mask'])
    out = dict(action_logits=action_logits, angle_logits=angle_logits.float())
    action_loss = F.cross_entropy(out['action_logits'], batch['action'])
    angle_valid = batch['action'] != model.action_count//2 if config.get('angle_nonstop_only',False) else torch.ones_like(batch['action'],dtype=torch.bool)
    angle_loss = (F.cross_entropy(out['angle_logits'][angle_valid], batch['angle_label'][angle_valid])
                  if angle_valid.any() else out['angle_logits'].sum()*0.)
    loss = config['action_loss_weight'] * action_loss + config['angle_loss_weight'] * angle_loss
    return loss, dict(action_loss=action_loss, angle_loss=angle_loss,
                      action_accuracy=(out['action_logits'].argmax(-1) == batch['action']).float().mean(),
                      angle_accuracy=(out['angle_logits'].argmax(-1) == batch['angle_label']).float().mean())
