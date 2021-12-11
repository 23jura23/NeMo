import pytorch_lightning as pl

from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="../conf/test", config_name="hifigan_test")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel.restore_from(cfg.checkpoint).eval()
    model.setup_validation_data(cfg.val_ds)
    model.set_trainer(trainer)
    trainer.validate(model)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
