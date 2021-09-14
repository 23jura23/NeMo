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

import argparse
import json
from pathlib import Path

from nemo.collections.nlp.models import PunctuationCapitalizationModel


"""
This script is for restoring punctuation and capitalization.

Usage example:

python punctuate_capitalize.py \
    --input_manifest <PATH_TO_INPUT_MANIFEST> \
    --output_manifest <PATH_TO_OUTPUT_MANIFEST>

<PATH_TO_INPUT_MANIFEST> is a path to NeMo ASR manifest. Usually it is an output of
    NeMo/examples/asr/transcribe_speech.py but can be a manifest with 'text' key. Alternatively you can use
    --input_text parameter for passing text for inference.
<PATH_TO_OUTPUT_MANIFEST> is a path to NeMo ASR manifest into which script output will be written. Alternatively
    you can use parameter --output_text.

For more details on this script usage look in argparse help.
"""


def get_args():
    default_model_parameter = "pretrained_name"
    default_model = "punctuation_en_bert"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="The script is for restoring punctuation and capitalization in text. Long strings are split into "
        "segments of length `--max_seq_length`. `--max_seq_length` is the length which includes [CLS] and [SEP] "
        "tokens. Parameter `--step` controls segments overlapping. `--step` is a distance between beginnings of "
        "consequent segments. Model outputs for tokens near the borders of tensors are less accurate and can be "
        "discarded before final predictions computation. Parameter `--margin` is number of discarded outputs near "
        "segments borders. Probabilities of tokens in overlapping parts of segments multiplied before selecting the "
        "best prediction. Default values of parameters `--max_seq_length`, `--step`, and `--margin` are optimal for "
        "IWSLT 2019 test dataset.",
    )
    input_ = parser.add_mutually_exclusive_group(required=True)
    input_.add_argument(
        "--input_manifest",
        "-m",
        type=Path,
        help="Path to the file with NeMo manifest which needs punctuation and capitalization. If the first element "
        "of manifest contains key 'pred_text', 'pred_text' values are passed for tokenization. Otherwise 'text' "
        "values are passed for punctuation and capitalization. Exactly one parameter of `--input_manifest` and "
        "`--input_text` should be provided.",
    )
    input_.add_argument(
        "--input_text",
        "-t",
        type=Path,
        help="Path to file with text which needs punctuation and capitalization. Exactly one parameter of "
        "`--input_manifest` and `--input_text` should be provided.",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output_manifest",
        "-M",
        type=Path,
        help="Path to output NeMo manifest. Text with restored punctuation and capitalization will be saved in "
        "'pred_text' elements if 'pred_text' key is present in the input manifest. Otherwise text with restored "
        "punctuation and capitalization will be saved in 'text' elements. Exactly one parameter of `--output_manifest` "
        "and `--output_text` should be provided.",
    )
    output.add_argument(
        "--output_text",
        "-T",
        type=Path,
        help="Path to file with text with restored punctuation and capitalization. Exactly one parameter of "
        "`--output_manifest` and `--output_text` should be provided.",
    )
    model = parser.add_mutually_exclusive_group(required=False)
    model.add_argument(
        "--pretrained_name",
        "-p",
        help=f"The name of NGC pretrained model. No more than one of parameters `--pretrained_name`, `--model_path`"
        f"should be provided. If neither of parameters `--pretrained_name` and `--model_path` are provided, then the "
        f"script is run with `--{default_model_parameter}={default_model}`.",
        choices=[m.pretrained_model_name for m in PunctuationCapitalizationModel.list_available_models()],
    )
    model.add_argument(
        "--model_path",
        "-P",
        type=Path,
        help=f"Path to .nemo checkpoint of punctuation and capitalization model. No more than one of parameters "
        f"`--pretrained_name` and `--model_path` should be provided. If neither of parameters `--pretrained_name` and "
        f"`--model_path` are provided, then the script is run with `--{default_model_parameter}={default_model}`.",
    )
    parser.add_argument(
        "--max_seq_length",
        "-L",
        type=int,
        default=64,
        help="Length of segments into which queries are split. `--max_seq_length` includes [CLS] and [SEP] tokens.",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=8,
        help="Relative shift of consequent segments into which long queries are split. Long queries are split into "
        "segments which can overlap. Parameter `step` controls such overlapping. Imagine that queries are "
        "tokenized into characters, `max_seq_length=5`, and `step=2`. In such a case query 'hello' is tokenized "
        "into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.",
    )
    parser.add_argument(
        "--margin",
        "-g",
        type=int,
        default=16,
        help="A number of subtokens in the beginning and the end of segments which output probabilities are not used "
        "for prediction computation. The first segment does not have left margin and the last segment does not have "
        "right margin. For example, if input sequence is tokenized into characters, `max_seq_length=5`, `step=1`, "
        "and `margin=1`, then query 'hello' will be tokenized into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], "
        "['[CLS]', 'e', 'l', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`. These segments are passed to the "
        "model. Before final predictions computation, margins are removed. In the next list, subtokens which logits "
        "are not used for final predictions computation are marked with asterisk: `[['[CLS]'*, 'h', 'e', 'l'*, "
        "'[SEP]'*], ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]`.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=128, help="Number of segments which are processed simultaneously.",
    )
    args = parser.parse_args()
    if args.input_manifest is None and args.output_manifest is not None:
        parser.error("--output_manifest requires --input_manifest")
    if args.pretrained_name is None and args.model_path is None:
        setattr(args, default_model_parameter, default_model)
    for name in ["input_manifest", "input_text", "output_manifest", "output_text", "model_path"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser())
    return args


def load_manifest(manifest: Path):
    result = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            result.append(data)
    return result


def main():
    args = get_args()
    if args.pretrained_name is None:
        model = PunctuationCapitalizationModel.restore_from(args.model_path)
    else:
        model = PunctuationCapitalizationModel.from_pretrained(args.pretrained_name)
    if args.input_manifest is None:
        texts = []
        with args.input_text.open() as f:
            texts.append(f.readline().strip())
    else:
        manifest = load_manifest(args.input_manifest)
        text_key = "pred_text" if "pred_text" in manifest[0] else "text"
        texts = []
        for item in manifest:
            texts.append(item[text_key])
    processed_texts = model.add_punctuation_capitalization(
        texts,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        step=args.step,
        margin=args.margin,
    )
    if args.output_manifest is None:
        with args.output_text.open('w') as f:
            for t in processed_texts:
                f.write(t + '\n')
    else:
        with args.output_manifest.open('w') as f:
            for item, t in zip(manifest, processed_texts):
                item[text_key] = t
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()
