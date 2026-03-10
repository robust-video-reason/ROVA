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
from typing import Any, Callable, Optional, Union, List, Tuple
from requests.exceptions import RequestException
import time
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from openai import OpenAI
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
data_dir = "/data2/lisida"

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
        openai_base_url: str = "https://api.nuwaapi.com/v1",
        openai_api_key: str = "sk-vqQD8ckOyfVysxhoJ0HfHd30K75KeJs1ctwz4WSYqiIixI01",
        openai_model: str = "gpt-4o",
        consistency_batch_size: int = 8,
        consistency_weight: float = 1.0,
        max_consistency_pairs_per_request: int = 8,
    ):
        # (The __init__ body is unchanged from your provided source except for adding kl_alpha default.)
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        if args.bf16:
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_init_kwargs["torch_dtype"] = torch.float16

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

        self.openai_client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
        self.openai_model = openai_model
        self.consistency_batch_size = consistency_batch_size
        self.consistency_weight = consistency_weight
        self.max_consistency_pairs_per_request = max_consistency_pairs_per_request

        self._consistency_system_prompt = (
            "You are an evaluator that compares two textual outputs. "
            "Return a JSON object EXACTLY in this format: "
            "{\"score\": <number between 0.0 and 1.0>, \"explanation\": \"<short text, <= 200 chars>\"}."
            " Score definition: 1.0 means candidate matches the reference in meaning, intent, and facts."
        )

        self.hard_questions = []
        self.easy_questions = []
        self.max_memory_size = getattr(script_args, "max_memory_size", 1024)
        self.train_time = 0.0
        self.eval_time = 0.0
        self.judge_gen_config = GenerationConfig(
            max_new_tokens=min(self.max_completion_length, 512),
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            num_return_sequences=max(1, self.num_generations // 4),
            pad_token_id=self.processing_class.pad_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
        )

    def save_memory(self, filepath: str):
        memory = {
            "hard_questions": self.hard_questions,
            "easy_questions": self.easy_questions,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

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
        with torch.no_grad():
            outputs = model(input_ids, **kwargs)
            logits = outputs.logits  # (B, L, V)
        return logits[:, :-1, :]
    
    def _get_completion_topk_logits(self, model, prompt_ids, completion_ids, k=16, **kwargs):
        """
        Return top-k logits for completion part only.
        
        Returns:
            topk_vals: (B, completion_len, k)
            topk_idx: (B, completion_len, k)
        """
        prompt_len = prompt_ids.size(1)
        completion_len = completion_ids.size(1)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, prompt_len + completion_len)
        
        with torch.no_grad():
            outputs = model(full_ids, **kwargs)
            logits = outputs.logits  # (B, total_len, V)

            # logits[:, prompt_len-1:prompt_len+completion_len-1, :] 对应 completion_ids 的预测
            completion_logits = logits[:, prompt_len-1:prompt_len+completion_len-1, :]  # (B, completion_len, V)
            
            # 提取top-k
            topk_vals, topk_idx = torch.topk(completion_logits, k, dim=-1)
        
        return topk_vals, topk_idx

    def compute_kl_from_logits(self, logits_p: torch.Tensor, logits_q: torch.Tensor, k: int = 64, token_mask: Optional[torch.Tensor] = None) -> list:
        """
        Compute per-example average KL(p || q) where p and q are raw logits.
        logits_p, logits_q: (batch, seq_len, vocab)
        token_mask: optional (batch, seq_len), 1 for real tokens, 0 for padding.
        Returns: list of floats (len=batch)
        """
        if logits_p.shape != logits_q.shape:
            raise ValueError("logits shapes must match")

        #p_logits = logits_p.float()
        #q_logits = logits_q.float()

        p_topk_vals, p_topk_idx = torch.topk(logits_p, k, dim=-1)  # (B, L, k)
        log_p_topk = F.log_softmax(p_topk_vals, dim=-1)            # (B, L, k)
        p_topk = log_p_topk.exp()

        # gather对应的q logits
        q_topk_vals = torch.gather(logits_q, -1, p_topk_idx)       # (B, L, k)
        log_q_topk = F.log_softmax(q_topk_vals, dim=-1)

        kl_token = torch.sum(p_topk * (log_p_topk - log_q_topk), dim=-1)  # (B, L)

        if token_mask is None:
            kl_per_example = kl_token.mean(dim=1)
        else:
            mask = token_mask.float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            kl_per_example = (kl_token * mask).sum(dim=1) / denom

        return [float(x) for x in kl_per_example.detach().cpu().numpy()]
    
    def compute_kl_from_dual_topk(
        self,
        p_topk_vals: torch.Tensor,  # (B, L, k)
        p_topk_idx: torch.Tensor,   # (B, L, k)
        q_topk_vals: torch.Tensor,  # (B, L, k)
        q_topk_idx: torch.Tensor,   # (B, L, k)
        token_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        双top-k的KL计算 (近似)
        
        假设: 两个分布的top-k能捕获大部分概率质量
        """
        batch_size, seq_len, k = p_topk_vals.shape
        device = p_topk_vals.device

        # 1. p的归一化概率
        log_p = F.log_softmax(p_topk_vals, dim=-1)  # (B, L, k)
        p_probs = log_p.exp()
        
        # 2. 构建q的完整分布估计
        # 对于p的每个top-k token，查找它在q中的logit
        # 使用broadcasting + masking
        p_idx_expanded = p_topk_idx.unsqueeze(-1)  # (B, L, k_p, 1)
        q_idx_expanded = q_topk_idx.unsqueeze(-2)  # (B, L, 1, k_q)
        matches = (p_idx_expanded == q_idx_expanded).float()  # (B, L, k_p, k_q)
        
        # 对于p中的每个token，找到q中对应的logit值
        q_vals_expanded = q_topk_vals.unsqueeze(-2)  # (B, L, 1, k_q)
        matched_q_vals = (matches * q_vals_expanded).sum(dim=-1)  # (B, L, k_p)
        
        # 对于没有匹配的token (不在q的top-k中)，赋予一个小值
        has_match = matches.sum(dim=-1) > 0  # (B, L, k_p)
        matched_q_vals = torch.where(
            has_match, 
            matched_q_vals, 
            torch.full_like(matched_q_vals, -20.0)  # 未匹配的给极小logit
        )
        
        # 3. 归一化q
        log_q = F.log_softmax(matched_q_vals, dim=-1)
        
        # 4. 计算KL
        kl_token = torch.sum(p_probs * (log_p - log_q), dim=-1)  # (B, L)
        
        if token_mask is None:
            kl_per_example = kl_token.mean(dim=1)
        else:
            mask = token_mask.float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            kl_per_example = (kl_token * mask).sum(dim=1) / denom
    
        return [float(x) for x in kl_per_example.detach().cpu().numpy()]

    def _apply_video_mask_to_prompt_inputs(self, masked_prompt_inputs: dict) -> Optional[dict]:
        """
        Apply token-level masking to Qwen2.5-VL processor outputs.
        Works with both pixel_values and pixel_values_videos (both are patch token embeddings).
        
        Args:
            prompt_inputs: Dictionary from Qwen2.5-VL processor containing:
                        - 'pixel_values': (num_tokens, feature_dim) for images/videos
                        - 'pixel_values_videos': (num_tokens, feature_dim) for videos
                        - other keys like 'input_ids', 'attention_mask', etc.
        
        Returns:
            masked_prompt_inputs: Deep copy of prompt_inputs with masked tokens,
                                or None if no visual inputs found or masking fails
        """
        # Check for visual inputs
        has_pixel_values = 'pixel_values' in masked_prompt_inputs
        has_pixel_values_videos = 'pixel_values_videos' in masked_prompt_inputs
        
        if not has_pixel_values and not has_pixel_values_videos:
            print("No visual inputs (pixel_values or pixel_values_videos) found in prompt_inputs")
            return None
        
        try:
            # Determine which key to process
            if has_pixel_values_videos:
                key_name = 'pixel_values_videos'
            else:
                key_name = 'pixel_values'
            
            pixel_values = masked_prompt_inputs[key_name]
            
            # Validate input type
            if not isinstance(pixel_values, torch.Tensor):
                print(f"Warning: {key_name} is not a torch.Tensor, got {type(pixel_values)}")
                return None
            
            # Log original info
            #print(f"   Original {key_name} shape: {pixel_values.shape}")
            #print(f"   dtype: {pixel_values.dtype}, device: {pixel_values.device}")
            #print(f"   Value range: [{pixel_values.min().item():.4f}, {pixel_values.max().item():.4f}]")
            
            # Apply token-level masking using the video_masker
            masked_tokens, masks = self.video_masker.mask_tokens(
                pixel_values,
                return_masks=True
            )
            
            # Update the prompt_inputs with masked tokens
            masked_prompt_inputs[key_name] = masked_tokens
            
            # Optionally store mask information
            if 'token_mask' in masks:
                masked_prompt_inputs['_mask_info'] = masks
                num_masked = int((masks['token_mask'] == 0).sum())
                total = len(masks['token_mask'])
                #print(f"Successfully masked {num_masked}/{total} tokens ({num_masked/total*100:.1f}%)")
            elif 'token_masks_batch' in masks:
                # Batch processing
                masked_prompt_inputs['_mask_info'] = masks
                total_masked = sum(
                    int((m['token_mask'] == 0).sum()) 
                    for m in masks['token_masks_batch']
                )
                total_tokens = sum(
                    len(m['token_mask']) 
                    for m in masks['token_masks_batch']
                )
                #print(f"Successfully masked {total_masked}/{total_tokens} tokens across batch")
            
            #print(f"Masked {key_name} shape: {masked_prompt_inputs[key_name].shape}")
            
            return masked_prompt_inputs
            
        except Exception as e:
            print(f"Error during token masking: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _mask_single_video_tensor(
        self, 
        video: torch.Tensor, 
        original_device: torch.device, 
        original_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Mask a single video tensor.
        
        Args:
            video: (T, C, H, W) tensor
            original_device: Original device of the tensor
            original_dtype: Original dtype of the tensor
        
        Returns:
            Masked video tensor in same format (T, C, H, W)
        """
        # Convert from (T, C, H, W) to (T, H, W, C) for masker
        video_thwc = video.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
        
        # Move to CPU and convert to numpy
        video_np = video_thwc.cpu().numpy()
        
        # Check if normalized to [0, 1] and convert to [0, 255] uint8 if needed
        was_normalized = False
        if video_np.max() <= 1.0 and video_np.min() >= 0.0:
            video_np = (video_np * 255.0).clip(0, 255).astype(np.uint8)
            was_normalized = True
        else:
            # Ensure uint8 format
            video_np = video_np.clip(0, 255).astype(np.uint8)
        
        masked_video_np, masks = self.video_masker.mask_video(
            video_np, 
            return_masks=False
        )
        
        masked_video_thwc = torch.from_numpy(masked_video_np)
        
        # Convert back to [0, 1] range if it was normalized
        if was_normalized:
            masked_video_thwc = masked_video_thwc.float() / 255.0
        else:
            masked_video_thwc = masked_video_thwc.float()
        
        masked_video_thwc = masked_video_thwc.to(original_dtype)
        
        # Convert from (T, H, W, C) back to (T, C, H, W)
        masked_video = masked_video_thwc.permute(0, 3, 1, 2)
        
        # Move back to original device
        masked_video = masked_video.to(original_device)
        
        return masked_video

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
    
    def _call_openai_chat(self, messages: List[dict], max_tokens: int = 400, retries: int = 3, backoff: float = 1.5):
        attempt = 0
        while True:
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return completion
            except RequestException as e:
                attempt += 1
                if attempt > retries:
                    raise
                time.sleep(backoff ** attempt)
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    raise
                time.sleep(backoff ** attempt)

    def _parse_openai_score(self, completion) -> Tuple[float, str]:
        try:
            # openai client: completion.choices[0].message.content
            text = completion.choices[0].message.content.strip()
        except Exception:
            try:
                text = completion.choices[0].text.strip()
            except Exception:
                return 0.0, "no-response"

        try:
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.0))
            explanation = str(parsed.get("explanation", "")).strip()
            score = max(0.0, min(1.0, score))
            return score, explanation
        except Exception:
            import re
            m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
            if m:
                val = float(m.group(1))
                val = max(0.0, min(1.0, val))
                rem = text.replace(m.group(0), "").strip()
                return val, rem[:200]
            return 0.0, "no-parse"

    def _compute_consistency_rewards_openai(self, references: List[str], candidates: List[str]) -> List[float]:
        assert len(references) == len(candidates)
        N = len(references)
        results: List[float] = []

        i = 0
        while i < N:
            end = min(i + self.max_consistency_pairs_per_request, N)
            block_refs = references[i:end]
            block_cands = candidates[i:end]
            user_msg = (
                "For each pair, produce a JSON array of objects with fields "
                '{"idx": <int>, "score": <0.0-1.0>, "explanation": "..."}.\n\nPairs:\n'
            )
            for j, (r, c) in enumerate(zip(block_refs, block_cands)):
                user_msg += f"IDX: {j}\nREFERENCE:\n{r}\n\nCANDIDATE:\n{c}\n\n---\n"

            messages = [
                {"role": "system", "content": self._consistency_system_prompt},
                {"role": "user", "content": user_msg},
            ]

            completion = self._call_openai_chat(messages, max_tokens=800)
            try:
                text = completion.choices[0].message.content.strip()
            except Exception:
                text = ""

            parsed_list = None
            try:
                parsed_list = json.loads(text)
            except Exception:
                parsed_list = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        parsed_list.append(obj)
                    except Exception:
                        continue

            for item in parsed_list:
                if not isinstance(item, dict):
                    results.append(0.0)
                    continue
                score = float(item.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                results.append(score)

            i = end

        if len(results) < N:
            results.extend([0.0] * (N - len(results)))
        return results[:N]

    def check_difficulty(
        self,
        question_text: str,
        image_inputs,
        masked_video_inputs,
        sample_item,
    ):
        """
        Evaluate whether the model can answer the question when only given MASKED video.

        Returns:
            "hard"      -> model consistently cannot answer
            "easy"      -> model consistently can answer
            "uncertain" -> mixed answers
        """

        # -------------------------------------------------------
        # 0. Prepare difficulty-check sample size (k)
        # -------------------------------------------------------
        k = max(1, self.num_generations // 4)  # or use self.shuffled_num_generations

        # -------------------------------------------------------
        # 1. Construct English judge prompt
        # -------------------------------------------------------
        judge_prompt = (
            "You are only allowed to view the masked version of the video. "
            "Using ONLY the masked video and no external assumptions, try to answer the question below.\n\n"
            "If you believe the masked video does NOT provide enough information to reliably answer, "
            "output exactly: 'NO'.\n"
            "If you believe you CAN answer based on the masked video, "
            "output exactly: 'YES'.\n\n"
            f"Question: {question_text}"
        )

        judge_prompts = [judge_prompt]

        # -------------------------------------------------------
        # 2. Tokenize masked input using processing_class
        # -------------------------------------------------------
        judge_inputs = self.processing_class(
            text=judge_prompts,
            images=image_inputs,
            videos=masked_video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        # Move tensors to correct device
        judge_inputs = {
            kname: kval.to(self.accelerator.device)
            for kname, kval in judge_inputs.items()
            if hasattr(kval, "to")
        }

        # -------------------------------------------------------
        # 3. Run generation with the model (no gradient)
        # -------------------------------------------------------
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with torch.no_grad():
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                output_ids = unwrapped_model.generate(
                    **judge_inputs,
                    generation_config=self.judge_gen_config,
                )

        # output_ids shape: [1*k, seq_len]
        prompt_ids = judge_inputs["input_ids"]
        plen = prompt_ids.size(1)
        completion_ids = output_ids[:, plen:]

        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        print(f"Difficulty check completions (k={k}): {completions}")

        # -------------------------------------------------------
        # 4. Refusal detection
        # -------------------------------------------------------
        def is_refusal(text: str) -> bool:
            # Use unified marker "UNANSWERABLE"
            text = text.strip().lower()
            if "unanswerable" in text:
                return True
            return False

        num_refuse = sum(is_refusal(c) for c in completions)

        all_cannot = (num_refuse == k)
        all_can = (num_refuse == 0)

        # -------------------------------------------------------
        # 5. Memory record
        # -------------------------------------------------------
        def append_memory(mem_list, item):
            mem_list.append(item)
            if len(mem_list) > self.max_memory_size:
                mem_list.pop(0)
        if all_cannot:
            append_memory(self.hard_questions, sample_item)
            return "hard"
        if all_can:
            append_memory(self.easy_questions, sample_item)
            return "easy"
        return "uncertain"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        start_time = time.time()
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        #prompts = [x["prompt"] for x in inputs]
        #prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompts = [inputs[0]["prompt"]]
        prompts_text = [maybe_apply_chat_template(inputs[0], self.processing_class)["prompt"]]

        input_copy = copy.deepcopy(inputs[0]['prompt'])
        input_copy = self.remove_none_from_data(input_copy)

        if inputs[0]['data_type'] == 'image':
            #input_copy[0]['content'][0]['image'] = data_dir + "/VSIBench" + inputs[0]['path'][1:]
            input_copy[0]['content'][0]['image'] = data_dir + "/Video-R1-data" + inputs[0]['path'][1:]
        elif inputs[0]['data_type'] == 'video':
            #input_copy[0]['content'][0]['video'] = data_dir + "/VSIBench" + inputs[0]['path'][1:]
            input_copy[0]['content'][0]['video'] = data_dir + "/Video-R1-data" + inputs[0]['path'][1:]

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)

        masked_video_inputs = None
        try:
            if video_inputs is not None:
                masked_for_list = []
                for vid in video_inputs:
                    if isinstance(vid, torch.Tensor):
                        if vid.ndim == 4:
                            t, c, h, w = vid.shape
                            vid_np = vid.permute(0, 2, 3, 1).cpu().numpy()
                            masked_np, masks = self.video_masker.mask_video(vid_np, return_masks=True)
                            masked_np = np.asarray(masked_np)
                            masked_tensor = torch.from_numpy(masked_np).permute(0, 3, 1, 2).to(vid.dtype).to(vid.device)
                            masked_for_list.append(masked_tensor)
                        elif vid.ndim == 5:
                            B, T, C, H, W = vid.shape
                            per_sample_masked = []
                            for b in range(B):
                                v_np = vid[b].permute(0, 2, 3, 1).cpu().numpy()
                                masked_np, masks = self.video_masker.mask_video(v_np, return_masks=True)
                                masked_np = np.asarray(masked_np)
                                masked_t = torch.from_numpy(masked_np).permute(0, 3, 1, 2).to(vid.dtype).to(vid.device)
                                per_sample_masked.append(masked_t)
                            masked_for_list.append(torch.stack(per_sample_masked, dim=0))
                        else:
                            try:
                                v_np = vid.cpu().numpy()
                                masked_np, masks = self.video_masker.mask_video(v_np, return_masks=True)
                                masked_np = np.asarray(masked_np)
                                if masked_np.ndim == 4:  # (T,H,W,C)
                                    masked_t = torch.from_numpy(masked_np).permute(0, 3, 1, 2).to(vid.dtype).to(vid.device)
                                else:
                                    masked_t = torch.from_numpy(masked_np).to(vid.dtype).to(vid.device)
                                masked_for_list.append(masked_t)
                            except Exception as e:
                                print(f"Warning: unable to mask video tensor with ndim={vid.ndim}: {e}")
                                masked_for_list.append(vid)
                    else:
                        try:
                            masked_out, masks = self.video_masker.mask_video(vid, return_masks=True)
                            if isinstance(masked_out, list):
                                masked_for_list.append(masked_out)
                            else:
                                m_np = np.asarray(masked_out)
                                if m_np.ndim == 4:
                                    masked_t = torch.from_numpy(m_np).permute(0, 3, 1, 2)
                                    masked_for_list.append(masked_t)
                                else:
                                    masked_for_list.append(masked_out)
                        except Exception as e:
                            print(f"Warning: video_masker failed on non-tensor video: {e}")
                            masked_for_list.append(vid)
                masked_video_inputs = masked_for_list if len(masked_for_list) > 0 else None
        except Exception as e:
            print(f"Masked video generation failed: {e}")
            masked_video_inputs = None

        start_eval_time = time.time()
        difficulty = self.check_difficulty(
            question_text=prompts_text[0],
            image_inputs=image_inputs,
            masked_video_inputs=masked_video_inputs,
            sample_item=inputs[0]
        )
        end_eval_time = time.time()
        self.eval_time += (end_eval_time - start_eval_time)

        if difficulty == "hard":
            loss = torch.zeros((), device=self.accelerator.device, requires_grad=True)
            return loss

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

        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

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
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]

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

        prompt_inputs_for_masking = copy.deepcopy(prompt_inputs)

        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
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
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

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

        if masked_video_inputs is not None:
            try:
                masked_prompt_inputs = self.processing_class(
                    text=copy.deepcopy(prompts_text),
                    images=image_inputs,
                    videos=masked_video_inputs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                masked_prompt_inputs = super()._prepare_inputs(masked_prompt_inputs)

                if "input_ids" in masked_prompt_inputs and self.max_prompt_length is not None:
                    masked_prompt_inputs["input_ids"] = masked_prompt_inputs["input_ids"][:, -self.max_prompt_length :]

                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    maskbranch_prompt_completion_ids = unwrapped_model.generate(**masked_prompt_inputs, generation_config=self.generation_config)

                prompt_length = prompt_ids.size(1)
                maskbranch_completion_ids = maskbranch_prompt_completion_ids[:, prompt_length:]
                maskbranch_completions = self.processing_class.batch_decode(maskbranch_completion_ids, skip_special_tokens=True)

                def _plain_text_from_decoded(x):
                    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], dict) and "content" in x[0]:
                        return x[0]["content"]
                    if isinstance(x, str):
                        return x
                    return str(x)

                plain_refs = [_plain_text_from_decoded(c) for c in completions]
                plain_cands = [_plain_text_from_decoded(c) for c in maskbranch_completions]

                # for c in plain_refs:
                #     print("completion:", c)
                # for c in plain_cands:
                #     print("mask_completion:", c)

                consistency_scores = self._compute_consistency_rewards_openai(plain_refs, plain_cands)
                consistency_tensor = torch.tensor(consistency_scores, dtype=torch.float32, device=self.accelerator.device)
                consistency_tensor = consistency_tensor.unsqueeze(1) * float(self.consistency_weight)
                rewards_per_func = torch.cat([rewards_per_func, consistency_tensor], dim=1)

            except Exception as e:
                print(f"Masked branch or consistency evaluation failed: {e}")

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
        if consistency_tensor is not None:
            self._metrics[f"rewards/consistency_reward"].append(reward_per_func[-1].item())

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

        end_time = time.time()
        self.train_time += (end_time - start_time) - (end_eval_time - start_eval_time)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def log_times(self):
        if not self.is_world_process_zero():
            return
        total_time = self.train_time + self.eval_time
        print(f"Total training time (excluding eval): {self.train_time:.2f} seconds")
        print(f"Total evaluation time: {self.eval_time:.2f} seconds")
        with open(os.path.join(self.args.output_dir, "time_log.txt"), "w") as f:
            f.write(f"Total training time (excluding eval): {self.train_time:.2f} seconds\n")
            f.write(f"Total evaluation time: {self.eval_time:.2f} seconds\n")
            f.write(f"Overall total time: {total_time:.2f} seconds\n")

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
