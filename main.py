import sys
import importlib
from easytorch import launch_training

if __name__ == "__main__":
    name = sys.argv[1]   # ví dụ: PEMS04.pretrain
    CFG = importlib.import_module(f"configs.{name}").CFG
    launch_training(CFG)