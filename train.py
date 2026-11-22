import argparse

#####
import debugpy

from clip.trainer.trainer import Trainer
from clip.utils.misc import load_config

debugpy.listen(("0.0.0.0", 6000))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
#####


def main(args):
    cfg = load_config(args.cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="clip/config/clip_config.yaml", help="cfg path")
    args = parser.parse_args()
    main(args)
