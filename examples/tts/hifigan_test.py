import pytorch_lightning as pl

from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf/hifigan", config_name="hifigan")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    trainer.validate(model)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
