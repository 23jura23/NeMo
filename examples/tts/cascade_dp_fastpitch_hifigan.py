import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from nemo.collections.tts.models.cascade_02 import Cascade02

@hydra_runner(config_path="conf/cascade02", config_name="cascade_dp_fastpitch_hifigan.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = Cascade02(cfg=cfg.model, trainer=trainer)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
