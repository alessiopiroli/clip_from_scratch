import argparse

from clip.evaluator.evaluator import Evaluator
from clip.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    evaluator = Evaluator(cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="clip/config/clip_config.yaml")
    args = parser.parse_args()
    main(args)
