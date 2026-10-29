import argparse
from easytorch import launch_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, required=True, 
                        help="Duong dan file cau hinh")
    parser.add_argument("--gpus", default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_training(args.cfg)