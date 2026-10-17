from habitat import Config
from ss_baselines.common.environments import AudioNavRLEnv
from .register import CollectRegister
import torch
def _load_hybrid_model(ckpt_path):
    from network import HybridNetwork

    device = torch.device('cuda')
    model = HybridNetwork().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

class COLLECTER:
    """
    collecter
    args:
        - greey collect
        - random collect
        - colliect collect
    """
    def __init__(self,config:Config,env:AudioNavRLEnv , **kwargs):
        self.env = env
        self.collect_config = config.TASK_CONFIG.COLLECT
        if self.collect_config.TYPE in ['greedy' , 'collided' , 'random' , 'angle' , 'offlineRL' , 'offlineRLtwoframe' , 'offlineLevel', 'offlineversion1' , 'offlineversion2' , 'offlineversion3' , 'offlineversion_2.0', 'offlineRL_v1_5', 'avwan_greedy']:
            cls = CollectRegister.get(self.collect_config.TYPE)
            
            if self.collect_config.TYPE in ['offlineRL' , 'offlineRLtwoframe' ,'collided']:
                model = _load_hybrid_model(kwargs['collect_ckpt'])
                self.collecter = cls(self.env , self.collect_config , model = kwargs['model'])
            elif self.collect_config.TYPE in ['offlineRL_v1_5']:
                self.collecter = cls(
                    self.env,
                    self.collect_config,
                    hybrid_model=kwargs['hybrid_model'],
                    offline_model=kwargs['offline_model'],
                )
            else:
                self.collecter = cls(self.env , self.collect_config)
        else:
            raise ValueError(f"collect name error!")
    def collect(self):
        self.collecter.collect()
