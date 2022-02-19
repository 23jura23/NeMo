import pytorch_lightning as pl

from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from hydra.utils import instantiate

@hydra_runner(config_path="../conf/test", config_name="hifigan_test")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    if cfg.pretrained and cfg.checkpoint is not None:
        raise ValueError("Cannot instantiate model from pretrained and checkpoint at the same time!")

    if cfg.pretrained:
        model = HifiGanModel.from_pretrained(model_name='tts_hifigan').eval()
#         model._cfg.validation_ds = cfg.val_ds
    else:
        model = HifiGanModel.restore_from(cfg.checkpoint).eval()    
    
#     print(model._cfg)
#     model.setup_training_data(cfg.val_ds)
    model.setup_validation_data(cfg.val_ds)
#     print(model._cfg)
    
#     print()
#     print(type(model._validation_dl.dataset))
    
    model.set_trainer(trainer)
    trainer.validate(model)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
