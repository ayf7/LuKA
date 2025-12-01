"""
CLI entrypoint for compressor training.

Parses arguments into config dataclasses and invokes the training loop. This
file is scaffolding only; no implementation is provided.
"""

from training.config import DataConfig, LossConfig, SamplerConfig, TrainConfig


def main():
    """
    Parse CLI arguments into configs and launch training.

    Args:
        None (uses sys.argv).

    Returns:
        None.

    Side effects:
        Should read command-line arguments, construct configs, and call
        training.trainer.train(). No logic is implemented in scaffolding.
    """
    raise NotImplementedError("CLI entrypoint is not implemented in scaffolding.")


if __name__ == "__main__":
    main()
