# copy license?

import itertools
from dataclasses import dataclass
from typing import Any, Dict
from typing import Union, Optional

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.data.datalayers import MelAudioDataset
from nemo.collections.tts.helpers.helpers import get_batch_size, get_num_workers
from nemo.collections.tts.helpers.helpers import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, MelLoss, PitchLoss
from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.collections.tts.torch.tts_data_types import SpeakerID
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal, LogprobsType
from nemo.core.neural_types.elements import (
    Index,
    LengthsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import CosineAnnealing, compute_max_steps
from nemo.utils import logging

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False



@dataclass
class FastPitchConfig:
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    input_fft: Dict[Any, Any] = MISSING
    output_fft: Dict[Any, Any] = MISSING
    duration_predictor: Dict[Any, Any] = MISSING
    pitch_predictor: Dict[Any, Any] = MISSING


# class FastPitchModel(SpectrogramGenerator, Exportable):
#     """FastPitch Model that is used to generate mel spectrograms from text"""
#
#     def __init__(self, cfg: DictConfig, trainer: Trainer = None):
#         if isinstance(cfg, dict):
#             cfg = OmegaConf.create(cfg)
#
#         self.learn_alignment = False
#         if "learn_alignment" in cfg:
#             self.learn_alignment = cfg.learn_alignment
#
#         self._normalizer = None
#         self._parser = None
#         self._tb_logger = None
#         super().__init__(cfg=cfg, trainer=trainer)
#
#         schema = OmegaConf.structured(FastPitchConfig)
#         # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
#         if isinstance(cfg, dict):
#             cfg = OmegaConf.create(cfg)
#         elif not isinstance(cfg, DictConfig):
#             raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
#         # Ensure passed cfg is compliant with schema
#         OmegaConf.merge(cfg, schema)
#
#         self.bin_loss_warmup_epochs = 100
#         self.log_train_images = False
#
#         loss_scale = 0.1 if self.learn_alignment else 1.0
#         dur_loss_scale = loss_scale
#         pitch_loss_scale = loss_scale
#         if "dur_loss_scale" in cfg:
#             dur_loss_scale = cfg.dur_loss_scale
#         if "pitch_loss_scale" in cfg:
#             pitch_loss_scale = cfg.pitch_loss_scale
#
#         self.mel_loss = MelLoss()
#         self.pitch_loss = PitchLoss(loss_scale=pitch_loss_scale)
#         self.duration_loss = DurationLoss(loss_scale=dur_loss_scale)
#
#         input_fft_kwargs = {}
#         self.aligner = None
#         if self.learn_alignment:
#             self.aligner = instantiate(self._cfg.alignment_module)
#             self.forward_sum_loss = ForwardSumLoss()
#             self.bin_loss = BinLoss()
#
#             self.ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]
#
#             if self.ds_class_name == "AudioToCharWithPriorAndPitchDataset":
#                 logging.warning(
#                     "AudioToCharWithPriorAndPitchDataset will be deprecated in 1.8 version. "
#                     "Please change your model to use Torch TTS Collection instead (e.g. see nemo.collections.tts.torch.data.TTSDataset)."
#                 )
#                 self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**self._cfg.train_ds.dataset.vocab)
#                 input_fft_kwargs["n_embed"] = len(self.vocab.labels)
#                 input_fft_kwargs["padding_idx"] = self.vocab.pad
#             elif self.ds_class_name == "TTSDataset":
#                 self.vocab = instantiate(self._cfg.train_ds.dataset.text_tokenizer)
#                 input_fft_kwargs["n_embed"] = len(self.vocab.tokens)
#                 input_fft_kwargs["padding_idx"] = self.vocab.pad
#             else:
#                 raise ValueError(f"Unknown dataset class: {self.ds_class_name}")
#
#         self.preprocessor = instantiate(self._cfg.preprocessor)
#
#         input_fft = instantiate(self._cfg.input_fft, **input_fft_kwargs)
#         output_fft = instantiate(self._cfg.output_fft)
#         duration_predictor = instantiate(self._cfg.duration_predictor)
#         pitch_predictor = instantiate(self._cfg.pitch_predictor)
#
#         self.fastpitch = FastPitchModule(
#             input_fft,
#             output_fft,
#             duration_predictor,
#             pitch_predictor,
#             self.aligner,
#             cfg.n_speakers,
#             cfg.symbols_embedding_dim,
#             cfg.pitch_embedding_kernel_size,
#             cfg.n_mel_channels,
#         )
#         self._input_types = self._output_types = None
#
#     @property
#     def tb_logger(self):
#         if self._tb_logger is None:
#             if self.logger is None and self.logger.experiment is None:
#                 return None
#             tb_logger = self.logger.experiment
#             if isinstance(self.logger, LoggerCollection):
#                 for logger in self.logger:
#                     if isinstance(logger, TensorBoardLogger):
#                         tb_logger = logger.experiment
#                         break
#             self._tb_logger = tb_logger
#         return self._tb_logger
#
#     @property
#     def normalizer(self):
#         if self._normalizer is not None:
#             return self._normalizer
#
#         if self.learn_alignment:
#             ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]
#
#             if ds_class_name == "AudioToCharWithPriorAndPitchDataset":
#                 logging.warning(
#                     "AudioToCharWithPriorAndPitchDataset will be deprecated in 1.8 version. "
#                     "Please change your model to use Torch TTS Collection instead (e.g. see nemo.collections.tts.torch.data.TTSDataset)."
#                 )
#                 self._normalizer = lambda x: x
#             elif ds_class_name == "TTSDataset":
#                 if "text_normalizer" not in self._cfg.train_ds.dataset:
#                     self._normalizer = lambda x: x
#                 else:
#                     normalizer = instantiate(self._cfg.train_ds.dataset.text_normalizer)
#                     text_normalizer_call = normalizer.normalize
#                     text_normalizer_call_args = {}
#                     if "text_normalizer_call_args" in self._cfg.train_ds.dataset:
#                         text_normalizer_call_args = self._cfg.train_ds.dataset.text_normalizer_call_args
#                     self._normalizer = lambda text: text_normalizer_call(text, **text_normalizer_call_args)
#             else:
#                 raise ValueError(f"Unknown dataset class: {ds_class_name}")
#         else:
#             # cfg.train_ds.dataset._target_ == "nemo.collections.asr.data.audio_to_text.FastPitchDataset"
#             self._normalizer = lambda x: x
#
#         return self._normalizer
#
#     @property
#     def parser(self):
#         if self._parser is not None:
#             return self._parser
#
#         if self.learn_alignment:
#             ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]
#
#             if ds_class_name == "AudioToCharWithPriorAndPitchDataset":
#                 logging.warning(
#                     "AudioToCharWithPriorAndPitchDataset will be deprecated in 1.8 version. "
#                     "Please change your model to use Torch TTS Collection instead (e.g. see nemo.collections.tts.torch.data.TTSDataset)."
#                 )
#                 if self.vocab is None:
#                     self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**self._cfg.train_ds.dataset.vocab)
#                 self._parser = self.vocab.encode
#             elif ds_class_name == "TTSDataset":
#                 tokenizer = instantiate(self._cfg.train_ds.dataset.text_tokenizer)
#                 self._parser = tokenizer.encode
#             else:
#                 raise ValueError(f"Unknown dataset class: {ds_class_name}")
#         else:
#             # cfg.train_ds.dataset._target_ == "nemo.collections.asr.data.audio_to_text.FastPitchDataset"
#             self._parser = parsers.make_parser(
#                 labels=self._cfg.labels,
#                 name='en',
#                 unk_id=-1,
#                 blank_id=-1,
#                 do_normalize=True,
#                 abbreviation_version="fastpitch",
#                 make_table=False,
#             )
#         return self._parser
#
#     def parse(self, str_input: str, normalize=True) -> torch.tensor:
#         if str_input[-1] not in [".", "!", "?"]:
#             str_input = str_input + "."
#
#         if normalize:
#             str_input = self.normalizer(str_input)
#
#         tokens = self.parser(str_input)
#
#         x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
#         return x
#
#     @typecheck(
#         input_types={
#             "text": NeuralType(('B', 'T_text'), TokenIndex()),
#             "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
#             "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
#             "speaker": NeuralType(('B'), Index()),
#             "pace": NeuralType(optional=True),
#             "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
#             "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
#             "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
#             "input_lens": NeuralType(('B'), LengthsType(), optional=True),
#         }
#     )
#     def forward(
#         self,
#         *,
#         text,
#         durs=None,
#         pitch=None,
#         speaker=0,
#         pace=1.0,
#         spec=None,
#         attn_prior=None,
#         mel_lens=None,
#         input_lens=None,
#     ):
#         return self.fastpitch(
#             text=text,
#             durs=durs,
#             pitch=pitch,
#             speaker=speaker,
#             pace=pace,
#             spec=spec,
#             attn_prior=attn_prior,
#             mel_lens=mel_lens,
#             input_lens=input_lens,
#         )
#
#     @typecheck(output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType())})
#     def generate_spectrogram(self, tokens: 'torch.tensor', speaker: int = 0, pace: float = 1.0) -> torch.tensor:
#         # FIXME: return masks as well?
#         self.eval()
#         spect, *_ = self(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace)
#         return spect
#
#     def training_step(self, batch, batch_idx):
#         attn_prior, durs, speaker = None, None, None
#         if self.learn_alignment:
#             if self.ds_class_name == "AudioToCharWithPriorAndPitchDataset":
#                 audio, audio_lens, text, text_lens, attn_prior, pitch, speaker = batch
#             elif self.ds_class_name == "TTSDataset":
#                 if SpeakerID in self._train_dl.dataset.sup_data_types_set:
#                     audio, audio_lens, text, text_lens, attn_prior, pitch, _, speaker = batch
#                 else:
#                     audio, audio_lens, text, text_lens, attn_prior, pitch, _ = batch
#             else:
#                 raise ValueError(f"Unknown vocab class: {self.vocab.__class__.__name__}")
#         else:
#             audio, audio_lens, text, text_lens, durs, pitch, speaker = batch
#
#         mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)
#
#         mels_pred, _, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur, pitch = self(
#             text=text,
#             durs=durs,
#             pitch=pitch,
#             speaker=speaker,
#             pace=1.0,
#             spec=mels if self.learn_alignment else None,
#             attn_prior=attn_prior,
#             mel_lens=spec_len,
#             input_lens=text_lens,
#         )
#         if durs is None:
#             durs = attn_hard_dur
#
#         mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
#         dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
#         loss = mel_loss + dur_loss
#         if self.learn_alignment:
#             ctc_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
#             bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
#             bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
#             loss += ctc_loss + bin_loss
#
#         pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
#         loss += pitch_loss
#
#         self.log("t_loss", loss)
#         self.log("t_mel_loss", mel_loss)
#         self.log("t_dur_loss", dur_loss)
#         self.log("t_pitch_loss", pitch_loss)
#         if self.learn_alignment:
#             self.log("t_ctc_loss", ctc_loss)
#             self.log("t_bin_loss", bin_loss)
#
#         # Log images to tensorboard
#         if self.log_train_images and isinstance(self.logger, TensorBoardLogger):
#             self.log_train_images = False
#
#             self.tb_logger.add_image(
#                 "train_mel_target",
#                 plot_spectrogram_to_numpy(mels[0].data.cpu().numpy()),
#                 self.global_step,
#                 dataformats="HWC",
#             )
#             spec_predict = mels_pred[0].data.cpu().numpy()
#             self.tb_logger.add_image(
#                 "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
#             )
#             if self.learn_alignment:
#                 attn = attn_hard[0].data.cpu().numpy().squeeze()
#                 self.tb_logger.add_image(
#                     "train_attn", plot_alignment_to_numpy(attn.T), self.global_step, dataformats="HWC",
#                 )
#                 soft_attn = attn_soft[0].data.cpu().numpy().squeeze()
#                 self.tb_logger.add_image(
#                     "train_soft_attn", plot_alignment_to_numpy(soft_attn.T), self.global_step, dataformats="HWC",
#                 )
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         attn_prior, durs, speaker = None, None, None
#         if self.learn_alignment:
#             if self.ds_class_name == "AudioToCharWithPriorAndPitchDataset":
#                 audio, audio_lens, text, text_lens, attn_prior, pitch, speaker = batch
#             elif self.ds_class_name == "TTSDataset":
#                 if SpeakerID in self._train_dl.dataset.sup_data_types_set:
#                     audio, audio_lens, text, text_lens, attn_prior, pitch, _, speaker = batch
#                 else:
#                     audio, audio_lens, text, text_lens, attn_prior, pitch, _ = batch
#             else:
#                 raise ValueError(f"Unknown vocab class: {self.vocab.__class__.__name__}")
#         else:
#             audio, audio_lens, text, text_lens, durs, pitch, speaker = batch
#
#         mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)
#
#         # Calculate val loss on ground truth durations to better align L2 loss in time
#         mels_pred, _, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur, pitch = self(
#             text=text,
#             durs=durs,
#             pitch=pitch,
#             speaker=speaker,
#             pace=1.0,
#             spec=mels if self.learn_alignment else None,
#             attn_prior=attn_prior,
#             mel_lens=mel_lens,
#             input_lens=text_lens,
#         )
#         if durs is None:
#             durs = attn_hard_dur
#
#         mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
#         dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
#         pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
#         loss = mel_loss + dur_loss + pitch_loss
#
#         return {
#             "val_loss": loss,
#             "mel_loss": mel_loss,
#             "dur_loss": dur_loss,
#             "pitch_loss": pitch_loss,
#             "mel_target": mels if batch_idx == 0 else None,
#             "mel_pred": mels_pred if batch_idx == 0 else None,
#         }
#
#     def validation_epoch_end(self, outputs):
#         collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
#         val_loss = collect("val_loss")
#         mel_loss = collect("mel_loss")
#         dur_loss = collect("dur_loss")
#         pitch_loss = collect("pitch_loss")
#         self.log("v_loss", val_loss)
#         self.log("v_mel_loss", mel_loss)
#         self.log("v_dur_loss", dur_loss)
#         self.log("v_pitch_loss", pitch_loss)
#
#         _, _, _, _, spec_target, spec_predict = outputs[0].values()
#
#         if isinstance(self.logger, TensorBoardLogger):
#             self.tb_logger.add_image(
#                 "val_mel_target",
#                 plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
#                 self.global_step,
#                 dataformats="HWC",
#             )
#             spec_predict = spec_predict[0].data.cpu().numpy()
#             self.tb_logger.add_image(
#                 "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
#             )
#             self.log_train_images = True
#
#     def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
#         if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
#             raise ValueError(f"No dataset for {name}")
#         if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
#             raise ValueError(f"No dataloder_params for {name}")
#         if shuffle_should_be:
#             if 'shuffle' not in cfg.dataloader_params:
#                 logging.warning(
#                     f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
#                     "config. Manually setting to True"
#                 )
#                 with open_dict(cfg.dataloader_params):
#                     cfg.dataloader_params.shuffle = True
#             elif not cfg.dataloader_params.shuffle:
#                 logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
#         elif not shuffle_should_be and cfg.dataloader_params.shuffle:
#             logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")
#
#         kwargs_dict = {}
#         if cfg.dataset._target_ == "nemo.collections.asr.data.audio_to_text.FastPitchDataset":
#             kwargs_dict["parser"] = self.parser
#         dataset = instantiate(cfg.dataset, **kwargs_dict)
#         return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)
#
#     def setup_training_data(self, cfg):
#         self._train_dl = self.__setup_dataloader_from_config(cfg)
#
#     def setup_validation_data(self, cfg):
#         self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="val")
#
#     def setup_test_data(self, cfg):
#         """Omitted."""
#         pass
#
#     @classmethod
#     def list_available_models(cls) -> 'List[PretrainedModelInfo]':
#         """
#         This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
#         Returns:
#             List of available pre-trained models.
#         """
#         list_of_models = []
#         model = PretrainedModelInfo(
#             pretrained_model_name="tts_en_fastpitch",
#             location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.4.0/files/tts_en_fastpitch_align.nemo",
#             description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
#             class_=cls,
#         )
#         list_of_models.append(model)
#
#         return list_of_models
#
#     ### Export code
#     def input_example(self):
#         """
#         Generates input examples for tracing etc.
#         Returns:
#             A tuple of input examples.
#         """
#         par = next(self.fastpitch.parameters())
#         inp = torch.randint(
#             0, self.fastpitch.encoder.word_emb.num_embeddings, (1, 44), device=par.device, dtype=torch.int64
#         )
#         pitch = torch.randn((1, 44), device=par.device, dtype=torch.float32) * 0.5
#         pace = torch.clamp((torch.randn((1, 44), device=par.device, dtype=torch.float32) + 1) * 0.1, min=0.01)
#
#         inputs = {'text': inp, 'pitch': pitch, 'pace': pace}
#
#         if self.fastpitch.speaker_emb is not None:
#             inputs['speaker'] = torch.randint(
#                 0, self.fastpitch.speaker_emb.num_embeddings, (1,), device=par.device, dtype=torch.int64
#             )
#
#         return (inputs,)
#
#     def forward_for_export(self, text, pitch, pace, speaker=None):
#         return self.fastpitch.infer(text=text, pitch=pitch, pace=pace, speaker=speaker)
#
#     @property
#     def input_types(self):
#         return self._input_types
#
#     @property
#     def output_types(self):
#         return self._output_types
#
#     def _prepare_for_export(self, **kwargs):
#         super()._prepare_for_export(**kwargs)
#
#         # Define input_types and output_types as required by export()
#         self._input_types = {
#             "text": NeuralType(('B', 'T_text'), TokenIndex()),
#             "pitch": NeuralType(('B', 'T_text'), RegressionValuesType()),
#             "pace": NeuralType(('B', 'T_text'), optional=True),
#             "speaker": NeuralType(('B'), Index()),
#         }
#         self._output_types = {
#             "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
#             "num_frames": NeuralType(('B'), TokenDurationType()),
#             "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
#             "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
#             "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
#         }
#
#     def _export_teardown(self):
#         self._input_types = self._output_types = None
#
#     @property
#     def disabled_deployment_input_names(self):
#         """Implement this method to return a set of input names disabled for export"""
#         disabled_inputs = set()
#         if self.fastpitch.speaker_emb is None:
#             disabled_inputs.add("speaker")
#         return disabled_inputs



class Cascade02(TextToWaveform, Exportable): # , Exportable
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        self.setup_training_data(cfg=cfg.fastpitch.model.train_ds)
        self.setup_validation_data(cfg=cfg.fastpitch.model.validation_ds)

        # fastpitch part

        self.learn_alignment = False
        if "learn_alignment" in cfg.fastpitch.model:
            self.learn_alignment = cfg.fastpitch.model.learn_alignment

        self._normalizer = None
        self._parser = None
        self._tb_logger = None

        schema = OmegaConf.structured(FastPitchConfig)
        # ModelPT ensures that cfg.fastpitch.model is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg.fastpitch.model, dict):
            cfg.fastpitch.model = OmegaConf.create(cfg.fastpitch.model)
        elif not isinstance(cfg.fastpitch.model, DictConfig):
            raise ValueError(f"cfg.fastpitch.model was type: {type(cfg.fastpitch.model)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg.fastpitch.model is compliant with schema
        OmegaConf.merge(cfg.fastpitch.model, schema)

        self.bin_loss_warmup_epochs = 100
        self.log_train_images = False

        loss_scale = 0.1 if self.learn_alignment else 1.0
        dur_loss_scale = loss_scale
        pitch_loss_scale = loss_scale
        if "dur_loss_scale" in cfg.fastpitch.model:
            dur_loss_scale = cfg.fastpitch.model.dur_loss_scale
        if "pitch_loss_scale" in cfg.fastpitch.model:
            pitch_loss_scale = cfg.fastpitch.model.pitch_loss_scale

        self.mel_loss = MelLoss()
        self.pitch_loss = PitchLoss(loss_scale=pitch_loss_scale)
        self.duration_loss = DurationLoss(loss_scale=dur_loss_scale)

        input_fft_kwargs = {}
        self.aligner = None
        if self.learn_alignment:
            self.aligner = instantiate(self._cfg.fastpitch.model.alignment_module)
            self.forward_sum_loss = ForwardSumLoss()
            self.bin_loss = BinLoss()

            self.ds_class_name = self._cfg.fastpitch.model.train_ds.dataset._target_.split(".")[-1]

            if self.ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                logging.warning(
                    "AudioToCharWithPriorAndPitchDataset will be deprecated in 1.8 version. "
                    "Please change your model to use Torch TTS Collection instead (e.g. see nemo.collections.tts.torch.data.TTSDataset)."
                )
                self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**self._cfg.fastpitch.model.train_ds.dataset.vocab)
                input_fft_kwargs["n_embed"] = len(self.vocab.labels)
                input_fft_kwargs["padding_idx"] = self.vocab.pad
            elif self.ds_class_name == "TTSDataset":
                self.vocab = instantiate(self._cfg.fastpitch.model.train_ds.dataset.text_tokenizer)
                input_fft_kwargs["n_embed"] = len(self.vocab.tokens)
                input_fft_kwargs["padding_idx"] = self.vocab.pad
            else:
                raise ValueError(f"Unknown dataset class: {self.ds_class_name}")

        self.preprocessor = instantiate(self._cfg.fastpitch.model.preprocessor)

        input_fft = instantiate(self._cfg.fastpitch.model.input_fft, **input_fft_kwargs)
        output_fft = instantiate(self._cfg.fastpitch.model.output_fft)
        duration_predictor = instantiate(self._cfg.fastpitch.model.duration_predictor)
        pitch_predictor = instantiate(self._cfg.fastpitch.model.pitch_predictor)

        self.fastpitch = FastPitchModule(
            input_fft,
            output_fft,
            duration_predictor,
            pitch_predictor,
            self.aligner,
            cfg.fastpitch.model.n_speakers,
            cfg.fastpitch.model.symbols_embedding_dim,
            cfg.fastpitch.model.pitch_embedding_kernel_size,
            cfg.fastpitch.model.n_mel_channels,
        )
        self._input_types = self._output_types = None

        # hifigan part

        self.audio_to_melspec_precessor = instantiate(cfg.hifigan.model.preprocessor)
        # use a different melspec extractor because:
        # 1. we need to pass grads
        # 2. we need remove fmax limitation
        self.trg_melspec_fn = instantiate(cfg.hifigan.model.preprocessor, highfreq=None, use_grads=True)
        self.generator = instantiate(cfg.hifigan.model.generator)
        self.mpd = MultiPeriodDiscriminator(debug=cfg.hifigan.model.debug if "debug" in cfg.hifigan.model else False)
        self.msd = MultiScaleDiscriminator(debug=cfg.hifigan.model.debug if "debug" in cfg.hifigan.model else False)
        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.l1_factor = cfg.hifigan.model.get("l1_loss_factor", 45)

        self.sample_rate = self._cfg.hifigan.model.preprocessor.sample_rate
        self.stft_bias = None

        if self._train_dl and isinstance(self._train_dl.dataset, MelAudioDataset):
            self.input_as_mel = True
        else:
            self.input_as_mel = False

        self.automatic_optimization = False

    # fastpitch functions

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    @property
    def normalizer(self):
        if self._normalizer is not None:
            return self._normalizer

        if self.learn_alignment:
            ds_class_name = self._cfg.fastpitch.model.train_ds.dataset._target_.split(".")[-1]

            if ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                logging.warning(
                    "AudioToCharWithPriorAndPitchDataset will be deprecated in 1.8 version. "
                    "Please change your model to use Torch TTS Collection instead (e.g. see nemo.collections.tts.torch.data.TTSDataset)."
                )
                self._normalizer = lambda x: x
            elif ds_class_name == "TTSDataset":
                if "text_normalizer" not in self._cfg.fastpitch.model.train_ds.dataset:
                    self._normalizer = lambda x: x
                else:
                    normalizer = instantiate(self._cfg.fastpitch.model.train_ds.dataset.text_normalizer)
                    text_normalizer_call = normalizer.normalize
                    text_normalizer_call_args = {}
                    if "text_normalizer_call_args" in self._cfg.fastpitch.model.train_ds.dataset:
                        text_normalizer_call_args = self._cfg.fastpitch.model.train_ds.dataset.text_normalizer_call_args
                    self._normalizer = lambda text: text_normalizer_call(text, **text_normalizer_call_args)
            else:
                raise ValueError(f"Unknown dataset class: {ds_class_name}")
        else:
            # cfg.train_ds.dataset._target_ == "nemo.collections.asr.data.audio_to_text.FastPitchDataset"
            self._normalizer = lambda x: x

        return self._normalizer

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        if self.learn_alignment:
            ds_class_name = self._cfg.fastpitch.model.train_ds.dataset._target_.split(".")[-1]

            if ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                logging.warning(
                    "AudioToCharWithPriorAndPitchDataset will be deprecated in 1.8 version. "
                    "Please change your model to use Torch TTS Collection instead (e.g. see nemo.collections.tts.torch.data.TTSDataset)."
                )
                if self.vocab is None:
                    self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**self._cfg.fastpitch.model.train_ds.dataset.vocab)
                self._parser = self.vocab.encode
            elif ds_class_name == "TTSDataset":
                tokenizer = instantiate(self._cfg.fastpitch.model.train_ds.dataset.text_tokenizer)
                self._parser = tokenizer.encode
            else:
                raise ValueError(f"Unknown dataset class: {ds_class_name}")
        else:
            # cfg.train_ds.dataset._target_ == "nemo.collections.asr.data.audio_to_text.FastPitchDataset"
            self._parser = parsers.make_parser(
                labels=self._cfg.fastpitch.model.labels,
                name='en',
                unk_id=-1,
                blank_id=-1,
                do_normalize=True,
                abbreviation_version="fastpitch",
                make_table=False,
            )
        return self._parser

    def parse(self, str_input: str, normalize=True) -> torch.tensor:
        if str_input[-1] not in [".", "!", "?"]:
            str_input = str_input + "."

        if normalize:
            str_input = self.normalizer(str_input)

        tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType())})
    def generate_spectrogram(self, tokens: 'torch.tensor', speaker: int = 0, pace: float = 1.0) -> torch.tensor:
        # FIXME: return masks as well?
        self.eval()
        spect, *_ = self.fastpitch(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace)
        return spect

    # hifigan functions

    def _get_max_steps(self):
        return compute_max_steps(
            max_epochs=self._cfg.trainer.max_epochs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            limit_train_batches=self.trainer.limit_train_batches,
            num_workers=get_num_workers(self.trainer),
            num_samples=len(self._train_dl.dataset),
            batch_size=get_batch_size(self._train_dl),
            drop_last=self._train_dl.drop_last,
        )

    def _get_warmup_steps(self, max_steps):
        warmup_steps = self._cfg.hifigan.model.sched.get("warmup_steps", None)
        warmup_ratio = self._cfg.hifigan.model.sched.get("warmup_ratio", None)

        if warmup_steps is not None and warmup_ratio is not None:
            raise ValueError(f'Either use warmup_steps or warmup_ratio for scheduler')

        if warmup_steps is not None:
            return warmup_steps

        if warmup_ratio is not None:
            return warmup_ratio * max_steps

        raise ValueError(f'Specify warmup_steps or warmup_ratio for scheduler')


    @typecheck(
        input_types={"spec": NeuralType(('B', 'C', 'T'), MelSpectrogramType())},
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def convert_spectrogram_to_audio(self, spec: 'torch.tensor') -> 'torch.tensor':
        return self.generator(spec=spec).squeeze(1)

    def _bias_denoise(self, audio, mel):
        def stft(x):
            comp = torch.stft(x.squeeze(1), n_fft=1024, hop_length=256, win_length=1024)
            real, imag = comp[..., 0], comp[..., 1]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase

        def istft(mags, phase):
            comp = torch.stack([mags * torch.cos(phase), mags * torch.sin(phase)], dim=-1)
            x = torch.istft(comp, n_fft=1024, hop_length=256, win_length=1024)
            return x

        # create bias tensor
        if self.stft_bias is None or self.stft_bias.shape[0] != audio.shape[0]:
            audio_bias = self(spec=torch.zeros_like(mel, device=mel.device))
            self.stft_bias, _ = stft(audio_bias)
            self.stft_bias = self.stft_bias[:, :, 0][:, :, None]

        audio_mags, audio_phase = stft(audio)
        audio_mags = audio_mags - self.cfg.get("denoise_strength", 0.0025) * self.stft_bias
        audio_mags = torch.clamp(audio_mags, 0.0)
        audio_denoised = istft(audio_mags, audio_phase).unsqueeze(1)

        return audio_denoised

    # common functions

    def configure_optimizers(self):
# here should go all of fastpitch modules with weights        self.optim_fp = instantiate(self._cfg.fastpitch.model.optim, params=itertools.chain(self.aligner.parameters(), self.fastpitch.parameters()))

        g_params = []

        if self.aligner is not None:
            g_params.append(self.aligner.parameters())

        g_params.append(self.fastpitch.parameters())
        g_params.append(self.generator.parameters())

        self.optim_g = instantiate(self._cfg.hifigan.model.optim, params=itertools.chain(*g_params))
        self.optim_d = instantiate(
            self._cfg.hifigan.model.optim, params=itertools.chain(self.msd.parameters(), self.mpd.parameters()),
        )

        if hasattr(self._cfg.hifigan.model, 'sched'):
            max_steps = self._cfg.hifigan.model.get("max_steps", None)
            if max_steps is None or max_steps < 0:
                max_steps = self._get_max_steps()

            warmup_steps = self._get_warmup_steps(max_steps)

            self.scheduler_g = CosineAnnealing(
                optimizer=self.optim_g, max_steps=max_steps, min_lr=self._cfg.hifigan.model.sched.min_lr, warmup_steps=warmup_steps,
            )  # Use warmup to delay start
            sch1_dict = {
                'scheduler': self.scheduler_g,
                'interval': 'step',
            }

            self.scheduler_d = CosineAnnealing(
                optimizer=self.optim_d, max_steps=max_steps, min_lr=self._cfg.hifigan.model.sched.min_lr,
            )
            sch2_dict = {
                'scheduler': self.scheduler_d,
                'interval': 'step',
            }

            return [self.optim_g, self.optim_d], [sch1_dict, sch2_dict]
        else:
            return [self.optim_g, self.optim_d]

    def convert_text_to_waveform(self, *, tokens: 'torch.tensor', **kwargs) -> 'List[torch.tensor]':
        with torch.no_grad:
            spec = self.generate_spectrogram(tokens)
            audio = self.convert_spectrogram_to_audio(spec)
            return audio

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "speaker": NeuralType(('B'), Index()),
            "pace": NeuralType(optional=True),
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
            "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
            "input_lens": NeuralType(('B'), LengthsType(), optional=True),
        }

    # fastpitch
    # input_types={
    #             "text": NeuralType(('B', 'T_text'), TokenIndex()),
    #             "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
    #             "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
    #             "speaker": NeuralType(('B'), Index()),
    #             "pace": NeuralType(optional=True),
    #             "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
    #             "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
    #             "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
    #             "input_lens": NeuralType(('B'), LengthsType(), optional=True),
    #         }
    # output_types={
    #             "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
    #             "num_frames": NeuralType(('B'), TokenDurationType()),
    #             "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
    #             "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
    #             "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
    #             "attn_soft": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
    #             "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
    #             "attn_hard": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
    #             "attn_hard_dur": NeuralType(('B', 'T_text'), TokenDurationType()),
    #             "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
    #         }
    #
    # hifigan
    # input_types={
    #             "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
    #         }
    # output_types={
    #             "audio": NeuralType(('B', 'S', 'T'), AudioSignal(self.sample_rate)),
    #         }


    @property
    def output_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "num_frames": NeuralType(('B'), TokenDurationType()),
            "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
            "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
            "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "attn_soft": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
            "attn_hard": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "attn_hard_dur": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "audio": NeuralType(('B', 'S', 'T_spec'), AudioSignal(self.sample_rate)),
        }

    @typecheck()
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=0,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
    ):
        """
        Runs fastpitch, then the generator, for inputs and outputs see input_types, and output_types
        """
        spect, \
        dec_lens, \
        durs_predicted, \
        log_durs_predicted, \
        pitch_predicted, \
        attn_soft, \
        attn_logprob, \
        attn_hard, \
        attn_hard_dur, \
        pitch = \
            self.fastpitch(
                text=text,
                durs=durs,
                pitch=pitch,
                speaker=speaker,
                pace=pace,
                spec=spec,
                attn_prior=attn_prior,
                mel_lens=mel_lens,
                input_lens=input_lens,
            )

        audio = self.generator(x=spect)

        return (spect,
                dec_lens,
                durs_predicted,
                log_durs_predicted,
                pitch_predicted,
                attn_soft,
                attn_logprob,
                attn_hard,
                attn_hard_dur,
                pitch,
                audio)

    # NOTE: audio, audio_lens could be incompatible with hifigan (hifigan uses AudioDataset, while FastPitch uses TTSDataset), if so we'll need to fix it
    def training_step(self, batch, batch_idx):
        attn_prior, durs, speaker = None, None, None
        if self.learn_alignment:
            if self.ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                audio, audio_lens, text, text_lens, attn_prior, pitch, speaker = batch
            elif self.ds_class_name == "TTSDataset":
                if SpeakerID in self._train_dl.dataset.sup_data_types_set:
                    audio, audio_lens, text, text_lens, attn_prior, pitch, _, speaker = batch
                else:
                    audio, audio_lens, text, text_lens, attn_prior, pitch, _ = batch  # this is my case
            else:
                raise ValueError(f"Unknown vocab class: {self.vocab.__class__.__name__}")
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)

        mels_pred, _, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur, pitch = self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            attn_prior=attn_prior,
            mel_lens=spec_len,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        loss = mel_loss + dur_loss
        if self.learn_alignment:
            ctc_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
            bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
            loss += ctc_loss + bin_loss

        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        loss += pitch_loss

        self.log("t_loss", loss)
        self.log("t_mel_loss", mel_loss)
        self.log("t_dur_loss", dur_loss)
        self.log("t_pitch_loss", pitch_loss)
        if self.learn_alignment:
            self.log("t_ctc_loss", ctc_loss)
            self.log("t_bin_loss", bin_loss)

        # Log images to tensorboard
        if self.log_train_images and isinstance(self.logger, TensorBoardLogger):
            self.log_train_images = False

            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(mels[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = mels_pred[0].data.cpu().numpy()
            self.tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )
            if self.learn_alignment:
                attn = attn_hard[0].data.cpu().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_attn", plot_alignment_to_numpy(attn.T), self.global_step, dataformats="HWC",
                )
                soft_attn = attn_soft[0].data.cpu().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_soft_attn", plot_alignment_to_numpy(soft_attn.T), self.global_step, dataformats="HWC",
                )

        # return loss



        # if in finetune mode the mels are pre-computed using a
        # spectrogram generator

        audio_mel = mels_pred
        audio_len = audio_lens

        # if self.input_as_mel:
        #     audio, audio_len, audio_mel = batch
        # # else, we compute the mel using the ground truth audio
        # else: # I think we'll be always here, no finetuning:
        #     audio, audio_len = batch
        #     # mel as input for generator
        #     audio_mel, _ = self.audio_to_melspec_precessor(audio, audio_len)

        # mel as input for L1 mel loss
        audio_trg_mel, _ = self.trg_melspec_fn(audio, audio_len)
        audio = audio.unsqueeze(1)

        # here we have 2 options: learn by fastpitch's mels, and by target mels (from the audio). I think we should choose 1 variant?

        #     исходное аудио             сгенерированное аудио
        #          |                   /                       \
        #      исходный мел           |                          \
        #         текст  - >  сгенерированный мел                мел сгенерированного аудио


        audio_pred = self.generator(x=audio_mel)
        audio_pred_mel, _ = self.trg_melspec_fn(audio_pred.squeeze(1), audio_len)

        # train discriminator
        self.optim_d.zero_grad()
        mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_pred.detach())
        loss_disc_mpd, _, _ = self.discriminator_loss(
            disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
        )
        msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_pred.detach())
        loss_disc_msd, _, _ = self.discriminator_loss(
            disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
        )
        loss_d = loss_disc_msd + loss_disc_mpd
        self.manual_backward(loss_d)
        self.optim_d.step()

        # train generator
        self.optim_g.zero_grad()
        loss_mel = F.l1_loss(audio_pred_mel, audio_trg_mel)
        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_pred)
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_pred)
        loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
        loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
        loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
        loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel * self.l1_factor

        loss_g += loss  # add fastpitch's loss to generator

        self.manual_backward(loss_g)
        self.optim_g.step()

        # run schedulers
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()

        metrics = {
            "g_loss_fm_mpd": loss_fm_mpd,
            "g_loss_fm_msd": loss_fm_msd,
            "g_loss_gen_mpd": loss_gen_mpd,
            "g_loss_gen_msd": loss_gen_msd,
            "g_loss": loss_g,
            "d_loss_mpd": loss_disc_mpd,
            "d_loss_msd": loss_disc_msd,
            "d_loss": loss_d,
            "global_step": self.global_step,
            "lr": self.optim_g.param_groups[0]['lr'],
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("g_l1_loss", loss_mel, prog_bar=True, logger=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        if self.input_as_mel:
            audio, audio_len, audio_mel = batch
            audio_mel_len = [audio_mel.shape[1]] * audio_mel.shape[0]
        else:
            audio, audio_len = batch
            audio_mel, audio_mel_len = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred = self(spec=audio_mel)

        # perform bias denoising
        pred_denoised = self._bias_denoise(audio_pred, audio_mel).squeeze(1)
        pred_denoised_mel, _ = self.audio_to_melspec_precessor(pred_denoised, audio_len)

        if self.input_as_mel:
            gt_mel, gt_mel_len = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)
        loss_mel = F.l1_loss(audio_mel, audio_pred_mel)

        self.log_dict({"val_loss": loss_mel}, on_epoch=True, sync_dist=True)

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            clips = []
            specs = []
            for i in range(min(5, audio.shape[0])):
                clips += [
                    wandb.Audio(
                        audio[i, : audio_len[i]].data.cpu().numpy(),
                        caption=f"real audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                    wandb.Audio(
                        audio_pred[i, 0, : audio_len[i]].data.cpu().numpy().astype('float32'),
                        caption=f"generated audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                    wandb.Audio(
                        pred_denoised[i, : audio_len[i]].data.cpu().numpy(),
                        caption=f"denoised audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                ]
                specs += [
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"input mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_pred_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"output mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(pred_denoised_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"denoised mel {i}",
                    ),
                ]
                if self.input_as_mel:
                    specs += [
                        wandb.Image(
                            plot_spectrogram_to_numpy(gt_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                            caption=f"gt mel {i}",
                        ),
                    ]

            self.logger.experiment.log({"audio": clips, "specs": specs})

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        return []

    # let's check if super implementation is still good for us
    # def load_state_dict(self, state_dict, strict=True):
    #     # override load_state_dict to give us some flexibility to be backward-compatible
    #     # with old checkpoints
    #     new_state_dict = {}
    #     num_resblocks = len(self.cfg['generator']['resblock_kernel_sizes'])
    #     for k, v in state_dict.items():
    #         new_k = k
    #         if 'resblocks' in k:
    #             parts = k.split(".")
    #             # only do this is the checkpoint type is older
    #             if len(parts) == 6:
    #                 layer = int(parts[2])
    #                 new_layer = f"{layer // num_resblocks}.{layer % num_resblocks}"
    #                 new_k = f"generator.resblocks.{new_layer}.{'.'.join(parts[3:])}"
    #         new_state_dict[new_k] = v
    #     super().load_state_dict(new_state_dict, strict=strict)





# class Cascade02(TextToWaveform):
#     def __init__(self, cfg: DictConfig, trainer: Trainer = None):
#         super().__init__(cfg, trainer)
#
#         if len(cfg.dp.preprocessing.languages) != 1:
#             raise ValueError(f"Only single language is supported, but got languages {cfg.dp.preprocessing.languages}")
#
#         self.lang = cfg.dp.preprocessing.languages[0]
#
#         if self.lang != 'en_us':
#             raise ValueError("Only english language is supported")
#
#         # но вообще просто стоит научиться язык внутрь передавать
#
#         # dict_dp_cfg = OmegaConf.to_container(cfg.dp, resolve=True, throw_on_missing=True)
#         # self.dp = create_dp_model(cfg.dp.model.type, dict_dp_cfg)
#         # preprocessor = Preprocessor.from_config(dict_dp_cfg)
#         # predictor = Predictor(model=self.dp, preprocessor=preprocessor)
#         # phonemizer = Phonemizer(predictor=predictor, lang_phoneme_dict=None) # TODO lang_phoneme_dict
#         # self.g2p_f = SingleLangPhonemizer(phonemizer, self.lang)
#
#         self.fastpitch = FastPitchModel(cfg=cfg.fastpitch.model
#                                         # , trainer=trainer
#                                         )
#
#         # self.fastpitch.vocab.g2p = self.g2p_f # and what happens to validation_ds?
#
#         self.hifigan = HifiGanModel(cfg=cfg.hifigan.model
#                                     # , trainer=trainer
#                                     )
#
#         self.automatic_optimization = False  # will see if needed at all
#
#
#     def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
#         pass
#
#     def convert_text_to_waveform(self, *, tokens: 'torch.tensor', **kwargs) -> 'List[torch.tensor]':
#         pass
#
#     @classmethod
#     def list_available_models(cls) -> 'List[PretrainedModelInfo]':
#         return []
#
#     def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
#         pass
#
#     def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
#         pass
#
#     def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
#         opt = self.optimizers()
#         opt.zero_grad()
#
#
#         fastpitch_loss, hifigan_loss = 0, 0
#
#         fastpitch_batch = batch['fastpitch']
#         hifigan_batch = batch['hifigan']
#
#         if fastpitch_batch is not None:
#             fastpitch_loss = self.fastpitch.training_step(fastpitch_batch, batch_idx)
#
#             # fastpitch_loss = self.fastpitch
#
#         if hifigan_batch is not None:
#             self.hifigan.training_step(hifigan_batch, batch_idx)
#
#         loss = fastpitch_loss
#
#         self.manual_backward(loss)  # or fastpitch.manual_backward(fastpitch_loss) ?
#
#         opt.step()
#
#         sched = self.lr_schedulers()
#         if sched is not None:
#             sched.step()
#
#
#         # return super().training_step(*args, **kwargs)
#
#     def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
#         return super().validation_step(*args, **kwargs)
#
# # def __init__(self, cfg: DictConfig, trainer: Trainer):
#     #     super().__init__()
#     #
#     #     self.fastpitch_config = OmegaConf.load(fastpitch_config_path)
#     #
#     #     self.fastpitch = instantiate(self.fastpitch_config['model'])
#     #
#     #     self.hifigan_config = OmegaConf.load(hifigan_config_path)
#     #
#     #     if g2p_type is None:
#     #         pass
#     #     elif g2p_type == "transformer":
#     #         pass
#     #     elif g2p_type == "dict":
#     #         pass
#     #     else:
#     #         raise ValueError(f"Unknown g2p_type: {g2p_type}")
#     #
#     # def forward(self):
#     #     pass