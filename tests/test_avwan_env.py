import sys
sys.path.append("./")
from experiment import AVWANRLEnv , get_avwan_config

def test_env(config):
    env = AVWANRLEnv(config= config)
    

if __name__ == "__main__":
    
    config = get_avwan_config()
    test_env(config)
