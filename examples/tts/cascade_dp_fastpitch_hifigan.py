import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

import importlib.util
import sys

file_path = '/home/jupyter/work/resources/utils/NeMo/nemo/collections/tts/models/cascade_02.py'
module_name = 'cascade_02'

spec = importlib.util.spec_from_file_location(module_name, file_path)
cascade_02 = importlib.util.module_from_spec(spec)
sys.modules[module_name] = cascade_02
spec.loader.exec_module(cascade_02)

print(cascade_02)

@hydra_runner(config_path="conf/cascade02", config_name="cascade_dp_fastpitch_hifigan.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.hifigan.exp_manager) # but need to merge exp_managers: different monitors
#     print(cfg.fastpitch.train_dataset)
    model = cascade_02.Cascade02(cfg=cfg, trainer=trainer)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
