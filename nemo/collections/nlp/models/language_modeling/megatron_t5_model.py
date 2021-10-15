# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
import re
import torch
from apex.transformer import tensor_parallel, parallel_state
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.t5_model import T5Model
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
    set_jit_fusion_options,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.parts.nlp_overrides import NLPNativeBfloat16PrecisionPlugin, NLPNativeMixedPrecisionPlugin
from nemo.utils import AppState, logging


class MegatronT5Model(NLPModel):
    """
    Megatron T5 pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg

        if self.cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=self.cfg.get('seed', 1234),
        )

        if not self.cfg.get('fused_bf16'):
            set_jit_fusion_options()

        self.tokenizer = get_nmt_tokenizer(
            library=self.cfg.tokenizer.library,
            model_name=self.cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer_model", self.cfg.tokenizer.model),
            vocab_file=self.register_artifact("vocab_file", self.cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("merges_file", self.cfg.tokenizer.merge_file),
        )
        self.num_sentinel_tokens = self.cfg.tokenizer.num_sentinel_tokens
        self._add_special_tokens_to_tokenizer()
        vocab_size = self.tokenizer.vocab_size

        padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=vocab_size,
            make_vocab_size_divisible_by=cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
        )

        self.model = T5Model(
            vocab_size=padded_vocab_size,
            hidden_size=cfg.hidden_size,
            max_position_embeddings=cfg.max_position_embeddings,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_attention_heads,
            apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=cfg.get('kv_channels', None),
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=cfg.get('pre_process', True),
            post_process=cfg.get('post_process', True),
            init_method_std=cfg.get('init_method_std', 0.02),
            fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=cfg.get('use_cpu_initialization', False),
            hidden_dropout=cfg.get('hidden_dropout', 0.1),
            fused_fp16=cfg.get('fused_fp16', False),
            fused_bf16=cfg.get('fused_bf16', False),
            fp32_residual_connection=cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
            bias_gelu_fusion=True,
            onnx_safe=cfg.get('onnx_safe', False),
        )

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None
    ):
        logits, encoder_hidden_states = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
            tokentype_ids,
            lm_labels,
            enc_hidden_states
        )
        return logits, encoder_hidden_states

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, \
            enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc,
            tokens_dec,
            enc_mask,
            dec_mask,
            enc_dec_mask,
            tokentype_ids=None,
            lm_labels=labels
        )

        loss = self.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        self.log('reduced_train_loss', averaged_loss[0], prog_bar=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, \
            enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc,
            tokens_dec,
            enc_mask,
            dec_mask,
            enc_dec_mask,
            tokentype_ids=None,
            lm_labels=labels
        )
        loss = self.loss_func(loss_mask, output_tensor)
        return loss

    def validation_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        self.log('val_loss', averaged_loss[0], prog_bar=True)

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_batch(self, batch):
        """Build the batch."""

        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask',
                'enc_mask', 'dec_mask', 'enc_dec_mask']
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = (data_b['enc_mask'] < 0.5)
        dec_mask = (data_b['dec_mask'] < 0.5)
        enc_dec_mask = (data_b['enc_dec_mask'] < 0.5)

        if self.cfg.debug:
            logging.info('debugging')
            tokens_enc_list = tokens_enc.detach().tolist()[0]
            tokens_dec_list = tokens_dec.detach().tolist()[0]
            labels_list = labels.detach().tolist()[0]
            logging.info(f'detokenize enc tokens: {self.tokenizer.ids_to_text(tokens_enc_list)}')
            logging.info(f'detokenize dec tokens: {self.tokenizer.ids_to_text(tokens_dec_list)}')
            logging.info(f'detokenize labels: {self.tokenizer.ids_to_text(labels_list)}')

        return tokens_enc, tokens_dec, loss_mask, labels, \
            enc_mask, dec_mask, enc_dec_mask

    def build_train_valid_test_datasets(self):
        logging.info('Building T5 datasets.')
        if self.cfg.data.seq_length_dec < self.cfg.data.seq_length * self.cfg.data.masked_lm_prob:
            raise ValueError(f"Cannot have decoder max sequence length ({self.cfg.data.seq_length_dec}) less than encoder sequence length ({self.cfg.data.seq_length}) * masked_lm_prob ({self.cfg.data.masked_lm_prob})")
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size / self.cfg.tensor_model_parallel_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            tokenizer=self.tokenizer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self.cfg.data.seq_length,
            max_seq_length_dec=self.cfg.data.seq_length_dec,
            masked_lm_prob=self.cfg.data.masked_lm_prob,
            short_seq_prob=self.cfg.data.short_seq_prob,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.skip_warmup,
            dataset_type='t5'
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T5 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if self.cfg.data.dataloader_type == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        elif self.cfg.data.dataloader_type == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            raise Exception('{} dataloader type is not supported.'.format(self.cfg.dataloader_type))

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            resume_checkpoint_path = self.trainer.checkpoint_connector.resume_checkpoint_path
            if resume_checkpoint_path:
                consumed_samples = int(
                    float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
                )
            else:
                consumed_samples = 0
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, global_step):
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return consumed_samples

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """PTL hook that is called after unscaling gradients when using native amp.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if isinstance(self.trainer.accelerator_connector.precision_plugin, NLPNativeMixedPrecisionPlugin):
            parameters = self.model.parameters()
            clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)
        else:
            return

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def _add_special_tokens_to_tokenizer(self):
        if self.cfg.tokenizer.library == 'huggingface' or self.cfg.tokenizer.library == 'megatron':
            additional_tokens = {
                'additional_special_tokens':
                [f'<extra_id_{i}>' for i in range(self.num_sentinel_tokens)]
            }
            self.tokenizer.add_special_tokens(additional_tokens)

        if self.cfg.tokenizer.library == 'sentencepiece':
            additional_tokens = [
                f'<extra_id_{i}>' for i in range(self.num_sentinel_tokens)
            ]
            self.tokenizer.add_special_tokens(additional_tokens)

    def list_available_models():
        pass
