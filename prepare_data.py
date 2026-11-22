import argparse

from clip.preprocessor.preprocessor import Preprocessor
from clip.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    preprocessor = Preprocessor(cfg)
    preprocessor.preprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="clip/config/clip_config.yaml", help="cfg path")
    args = parser.parse_args()
    main(args)
