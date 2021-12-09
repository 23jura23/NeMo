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

from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_tuning_dataset import GPTPromptTuningDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import IndexedDataset, MMapIndexedDataset
# TODO: refactor these datasets to work without megatron-lm dependency
# from nemo.collections.nlp.data.language_modeling.megatron.bert_dataset import BertDataset
# from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset
