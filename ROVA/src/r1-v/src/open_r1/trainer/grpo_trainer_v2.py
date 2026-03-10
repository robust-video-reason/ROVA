# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random
import copy
import math

import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    (Docstring kept from original - omitted for brevity)
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # (The __init__ body is unchanged from your provided source except for adding kl_alpha default.)
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
        else:
            pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1,
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,
            temperature=1,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # KL/Mask hyperparameter (default)
        self.kl_alpha = getattr(script_args, "kl_alpha", 1.0) if script_args is not None else 1.0
        # video_masker may be attached externally (see grpo.py). Default None.
        self.video_masker = getattr(self, "video_masker", None)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _get_full_logits(self, model, input_ids, **kwargs):
        """
        Return full logits aligned with input_ids (exclude the last logit so logits.shape[1] == input_ids.shape[1]-1)
        Output shape: (B, L-1, V)
        """
        outputs = model(input_ids, **kwargs)
        logits = outputs.logits  # (B, L, V)
        return logits[:, :-1, :]

    def compute_kl_from_logits(self, logits_p: torch.Tensor, logits_q: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> list:
        """
        Compute per-example average KL(p || q) where p and q are raw logits.
        logits_p, logits_q: (batch, seq_len, vocab)
        token_mask: optional (batch, seq_len), 1 for real tokens, 0 for padding.
        Returns: list of floats (len=batch)
        """
        if logits_p.shape != logits_q.shape:
            raise ValueError("logits shapes must match")

        p_logits = logits_p.float()
        q_logits = logits_q.float()

        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        p = torch.exp(log_p)

        kl_token = torch.sum(p * (log_p - log_q), dim=-1)  # (B, L)
        if token_mask is None:
            kl_per_example = kl_token.mean(dim=1)
        else:
            mask = token_mask.float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            kl_per_example = (kl_token * mask).sum(dim=1) / denom

        return [float(x) for x in kl_per_example.detach().cpu().numpy()]

    def _apply_video_mask_to_prompt_inputs(self, prompt_inputs: dict) -> Optional[dict]:
        """
        Try to apply self.video_masker to prompt_inputs in-place and return masked_prompt_inputs.
        If cannot apply (no video/pixel keys or masking fails), return None.
        NOTE: This is conservative — it tries to find likely keys and per-example mask them.
        """
        if self.video_masker is None:
            return None

        masked_inputs = copy.deepcopy(prompt_inputs)

        # Common keys that may contain video tensors
        candidate_keys = []
        if "pixel_values_videos" in masked_inputs:
            candidate_keys.append("pixel_values_videos")
        if "pixel_values" in masked_inputs:
            candidate_keys.append("pixel_values")
        # If no pixel keys, nothing to do
        if not candidate_keys:
            return None

        applied = False
        device = next(self.model.parameters()).device

        for key in candidate_keys:
            try:
                tensor = masked_inputs[key]
                # Expect tensor to be batch-first: (B, ...) ; iterate per example
                if not torch.is_tensor(tensor):
                    continue
                B = tensor.shape[0]
                masked_per_example = []
                for i in range(B):
                    single = tensor[i]
                    # Move to cpu numpy/torch for masker (masker accepts torch.Tensor)
                    # video_masker.mask_video expects (T,H,W,C) or similar per-example video tensor
                    try:
                        # ensure single is detached cpu
                        single_cpu = single.detach().cpu()
                        # pass single_cpu to mask_video (VideoMasker handles torch tensors)
                        masked_single, masks = self.video_masker.mask_video(single_cpu)
                        # masked_single may be torch or numpy depending on masker; ensure torch on original device
                        if isinstance(masked_single, torch.Tensor):
                            masked_single = masked_single.to(device)
                        else:
                            masked_single = torch.from_numpy(masked_single).to(device)
                        masked_per_example.append(masked_single)
                    except Exception:
                        # If per-example masking fails, bail out for this key
                        masked_per_example = None
                        break
                if masked_per_example is None:
                    continue
                # Stack back
                new_tensor = torch.stack(masked_per_example, dim=0)
                masked_inputs[key] = new_tensor
                applied = True
                break
            except Exception:
                continue

        if not applied:
            return None

        # ensure tensors are on device and prepared
        masked_inputs = super()._prepare_inputs(masked_inputs)
        return masked_inputs

    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        input_copy = copy.deepcopy(inputs[0]['prompt'])
        input_copy = self.remove_none_from_data(input_copy)

        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = os.getcwd() + "/data" + inputs[0]['path'][1:]
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = os.getcwd() + "/data" + inputs[0]['path'][1:]

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)

        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        # Prepare and store a copy for later masked attempt
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs_for_masking = copy.deepcopy(prompt_inputs)

        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length : ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length : ]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length : ]
            prompt_mask = prompt_mask[:, -self.max_prompt_length : ]

        if self.temporal and video_inputs:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length : ]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length : ]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

            if self.temporal:
                if video_inputs:
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                    shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                    shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
                else:
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.dummy_generation_config)

        # Prepare completion masking (after first EOS)
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # remove ids from prompt_inputs; keep pixel/video tensors for logits forward
        prompt_inputs.pop("input_ids", None)
        prompt_inputs.pop("attention_mask", None)

        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)

        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]

        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 : ]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)

        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 : ]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 : ]

        # Compute the per-token KL between model and reference (existing behavior)
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        # If temporal & video, handle shuffled branch rewards
        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]

            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)

        # Decode generated completions and compute base rewards
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # ------------------ KL consistency w.r.t masked video (MODIFIED: orig -> no_grad for reward; masked -> differentiable loss) ------------------
        kl_reward_vals = None
        masked_kl_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        try:
            orig_logits = None
            masked_logits = None
            # compute orig_logits under no_grad (original video -> only used for reward, no gradient)
            try:
                with torch.no_grad():
                    orig_logits = self._get_full_logits(model, prompt_completion_ids, **prompt_inputs)
            except Exception as e:
                orig_logits = None

            masked_prompt_inputs = self._apply_video_mask_to_prompt_inputs(prompt_inputs_for_masking)
            if masked_prompt_inputs is not None and orig_logits is not None:
                try:
                    # repeat masked pixel/video tensors to match prompt_completion_ids batch
                    if inputs[0]['data_type'] == 'image':
                        if "pixel_values" in masked_prompt_inputs:
                            masked_prompt_inputs["pixel_values"] = masked_prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
                            if "image_grid_thw" in masked_prompt_inputs:
                                masked_prompt_inputs["image_grid_thw"] = masked_prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
                    if inputs[0]['data_type'] == 'video':
                        if "pixel_values_videos" in masked_prompt_inputs:
                            masked_prompt_inputs["pixel_values_videos"] = masked_prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
                        if "video_grid_thw" in masked_prompt_inputs:
                            masked_prompt_inputs["video_grid_thw"] = masked_prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
                            if 'second_per_grid_ts' in masked_prompt_inputs:
                                del masked_prompt_inputs["second_per_grid_ts"]
                except Exception:
                    pass

                # compute masked_logits normally (keep autograd) so masked branch produces gradients
                try:
                    masked_logits = self._get_full_logits(model, prompt_completion_ids, **masked_prompt_inputs)
                except Exception as e:
                    masked_logits = None

            if orig_logits is not None and masked_logits is not None:
                token_mask = completion_mask
                # 1) compute detached KL values -> reward (keep original behavior: detach -> float -> math.exp)
                try:
                    kl_values = self.compute_kl_from_logits(orig_logits.detach(), masked_logits.detach(), token_mask=token_mask)
                except Exception:
                    # fallback approximate per-sample KL (detached)
                    with torch.no_grad():
                        orig_log_probs = torch.log_softmax(orig_logits.detach(), dim=-1)
                        masked_log_probs = torch.log_softmax(masked_logits.detach(), dim=-1)
                        orig_probs = torch.softmax(orig_logits.detach(), dim=-1)
                        kl_per_token = (orig_probs * (orig_log_probs - masked_log_probs)).sum(-1)  # (B*G, L)
                        kl_values = (kl_per_token * token_mask).sum(dim=1) / (token_mask.sum(dim=1).clamp_min(1))

                alpha = getattr(self, "kl_alpha", 1.0)
                kl_reward_vals = torch.tensor([math.exp(-alpha * float(k)) for k in kl_values], dtype=torch.float32, device=device)

                # 2) build differentiable masked-kl loss: treat orig as fixed, masked as variable
                # orig_probs detached -> no grad; masked_log_probs keeps grad
                orig_probs_fixed = torch.softmax(orig_logits.detach(), dim=-1)
                masked_log_probs = torch.log_softmax(masked_logits, dim=-1)
                kl_per_token_tensor = (orig_probs_fixed * (torch.log(orig_probs_fixed + 1e-12) - masked_log_probs)).sum(-1)  # (B*G, L)
                denom = token_mask.sum(dim=1).clamp_min(1).to(kl_per_token_tensor.dtype)
                masked_kl_per_sample = (kl_per_token_tensor * token_mask).sum(dim=1) / denom
                masked_kl_loss = masked_kl_per_sample.mean()
                masked_kl_coef = getattr(self, "masked_kl_coef", 1.0)
                masked_kl_loss = masked_kl_coef * masked_kl_loss

                # add kl rewards to the aggregated rewards (simple additive fusion) -- keep original reward behavior
                rewards = rewards_per_func.sum(dim=1)
                rewards = rewards + kl_reward_vals
            else:
                rewards = rewards_per_func.sum(dim=1)
        except Exception as e:
            print(f"KL masked branch failed: {e}")
            rewards = rewards_per_func.sum(dim=1)

        # If masked branch did not run but rewards not set above, set it here
        if 'rewards' not in locals():
            rewards = rewards_per_func.sum(dim=1)

        # Temporal adjustment and length control (as original)
        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()

            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask_sel = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask_sel, 0] = temporal_rewards_per_func[mask_sel, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to(device)
            else:
                temporal_rewards = torch.tensor([0.0]).to(device)
        else:
            temporal_rewards = torch.tensor([0.5]).to(device)

        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask_sel = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask_sel, as_tuple=True)[0].tolist()
            if len(selected_indices) > 1:
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2

        # Compute grouped-wise statistics and advantages (same as original)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Add differentiable masked-KL loss so the masked-video path has gradients
        if isinstance(masked_kl_loss, torch.Tensor) and masked_kl_loss.item() != 0.0:
            loss = loss + masked_kl_loss

        # Logging metrics (keeps original behavior)
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        num_devices = gathered_rewards.size(0) // self.num_generations
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices

        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices

        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)

        if self.temporal:
            temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
            self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards_list).mean().item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


class Qwen2VLGRPOVLLMTrainerModified(Qwen2VLGRPOTrainer):
    pass
