import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from video_mask import MaskConfig, VideoMasker

@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'kl'"},
    )
    max_pixels: Optional[int] = field(
        default=200704,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    max_frames: Optional[int] = field(
        default=16,
        metadata={"help": "Maximum number of frames for the video"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    kl_alpha: Optional[float] = field(
        default=3.0,
        metadata={"help": "alpha factor used in mapping KL -> reward (reward = exp(-alpha * KL))."},
    )
    kl_reward_name: Optional[str] = field(
        default="kl_consistency",
        metadata={"help": "Name to register KL reward function as."},
    )
    # ========= VideoMasker =========
    photometric_prob: Optional[float] = field(
        default=0.25,
        metadata={"help": "Probability of applying photometric (lighting) effects to a video."},
    )
    weather_prob: Optional[float] = field(
        default=0.25,
        metadata={"help": "Probability of applying weather effects to a video."},
    )
    occlusion_prob: Optional[float] = field(
        default=0.25,
        metadata={"help": "Probability of applying occlusion (block mask) to a video."},
    )
    shake_prob: Optional[float] = field(
        default=0.25,
        metadata={"help": "Probability of applying camera shake to a video."},
    )

    occlusion_mask_ratio: Optional[float] = field(
        default=0.3,
        metadata={"help": "Fraction of area to occlude when occlusion is chosen."},
    )
    occlusion_block_mean: Optional[int] = field(
        default=64,
        metadata={"help": "Mean block size (pixels) for occlusion blocks."},
    )
    occlusion_block_std: Optional[float] = field(
        default=16.0,
        metadata={"help": "Std of block size (pixels) for occlusion blocks."},
    )
    mask_value: Optional[float] = field(
        default=0.0,
        metadata={"help": "Value to fill masked pixels/frames with."},
    )

    enable_temporal_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to enable temporal masking in VideoMasker."},
    )
    frame_mask_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "Fraction of frames to mask/drop."},
    )
    temporal_mode: Optional[str] = field(
        default="random_drop",
        metadata={"help": "random_drop / drop_segments / keep_k"},
    )
    temporal_segment_len: Optional[int] = field(
        default=4,
        metadata={"help": "Segment length when temporal_mode == drop_segments"},
    )
    keep_k_frames: Optional[int] = field(
        default=None,
        metadata={"help": "If temporal_mode == keep_k, how many frames to keep"},
    )

    lighting_type: Optional[str] = field(
        default="random",
        metadata={"help": "Lighting type: dusk / night / overexposure / shadows / random."},
    )
    lighting_intensity: Optional[float] = field(
        default=0.7,
        metadata={"help": "Intensity of lighting effect."},
    )

    weather_type: Optional[str] = field(
        default="random",
        metadata={"help": "Weather type: light_rain / heavy_rain / storm / snow / hail / random."},
    )
    weather_particle_density: Optional[float] = field(
        default=0.5,
        metadata={"help": "Density (intensity) of weather particles."},
    )
    weather_particle_size: Optional[int] = field(
        default=2,
        metadata={"help": "Size of weather particles."},
    )
    weather_speed: Optional[int] = field(
        default=8,
        metadata={"help": "Speed of weather particles."},
    )
    weather_effect_intensity: Optional[float] = field(
        default=0.7,
        metadata={"help": "Overall intensity of weather effects."},
    )

    shake_intensity: Optional[float] = field(
        default=0.02,
        metadata={"help": "Camera shake intensity."},
    )
    zoom_min: Optional[float] = field(
        default=0.95,
        metadata={"help": "Minimum zoom factor for camera shake."},
    )
    zoom_max: Optional[float] = field(
        default=1.05,
        metadata={"help": "Maximum zoom factor for camera shake."},
    )
    rotation_min: Optional[float] = field(
        default=-2.0,
        metadata={"help": "Minimum rotation (degrees) for camera shake."},
    )
    rotation_max: Optional[float] = field(
        default=2.0,
        metadata={"help": "Maximum rotation (degrees) for camera shake."},
    )
    smoothness: Optional[float] = field(
        default=0.1,
        metadata={"help": "Smoothness of camera motion."},
    )


# ---------------- reward functions ----------------
def accuracy_reward(completions, solution, **kwargs):
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    question_type = kwargs.get('problem_type', [None])[0] if 'problem_type' in kwargs else None

    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "grpo_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def kl_consistency_reward(completions, **kwargs):
    """
    Expects kwargs to contain 'kl_values' -> list[float] corresponding to samples in this batch.
    Maps KL -> reward via exp(-alpha * KL), alpha provided via kwargs.get('kl_alpha', 1.0)
    If kl_values missing, returns zeros.
    """
    kl_values = kwargs.get("kl_values", None)
    alpha = kwargs.get("kl_alpha", 1.0)
    if kl_values is None:
        # no kl info available: return zeros (or you may want to return neutral 0.5)
        print("kl_consistency_reward: kl_values missing, returning zero rewards")
        return [0.0 for _ in range(len(completions))]
    # ensure same length
    if len(kl_values) != len(completions):
        print("kl_consistency_reward: kl_values length mismatch, returning zero rewards")
        # fallback: zeros
        return [0.0 for _ in range(len(completions))]
    rewards = []
    for kl in kl_values:
        try:
            reward = float(np.exp(-alpha * float(kl)))
        except Exception:
            print("Error computing kl_consistency_reward, setting reward=0.0")
            reward = 0.0
        rewards.append(reward)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# ------------------ KL helpers ------------------
def compute_kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> List[float]:
    """
    Compute per-example average KL(p || q) where p and q are model logits.
    logits_p, logits_q: (batch, seq_len, vocab)
    token_mask: optional (batch, seq_len), 1 for real tokens, 0 for padding.
    Returns: list of KL floats (len=batch).
    Numerically stable: uses log_softmax + softmax.
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError("logits shapes must match")
    # ensure float
    p_logits = logits_p.float()
    q_logits = logits_q.float()

    # compute log probs
    log_p = F.log_softmax(p_logits, dim=-1)  # (B, L, V)
    log_q = F.log_softmax(q_logits, dim=-1)
    p = torch.exp(log_p)  # probabilities

    # elementwise KL per token: sum_v p * (log_p - log_q)
    # shape (B, L)
    kl_token = torch.sum(p * (log_p - log_q), dim=-1)

    if token_mask is None:
        # average over seq_len
        kl_per_example = kl_token.mean(dim=1)  # (B,)
    else:
        # mask: only average over valid tokens
        mask = token_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        kl_per_example = (kl_token * mask).sum(dim=1) / denom

    # convert to python floats
    return [float(x) for x in kl_per_example.detach().cpu().numpy()]

# ------------------ Dataset / mapping helpers ------------------
def _load_video_from_example(example: Dict[str, Any]):
    """
    Try to load video frames from example.
    Two common possibilities:
      - example has 'frames' field which is list/ndarray/torch tensor
      - example has 'path' field which points to a video file (not handled here)
    If cannot load, returns None.
    NOTE: Do not assume heavy IO here; if your dataset only stores paths, prefer to do masking at __getitem__ time.
    """
    if "frames" in example and example["frames"] is not None:
        return example["frames"]
    # if there is a path field, user may want to load via external libs (torchvision / decord / av). Keep simple:
    if "path" in example and example["path"]:
        # not implementing video file loading here to avoid extra deps; user can customize
        return None
    return None

# ------------------ Main ------------------
def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # load dataset (support json / jsonl / named HF datasets)
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image_and_video(example):
        if example.get("problem_type") == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example.get("options", []):
                question += op + "\n"
        else:
            question = example['problem']

        msg = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": example.get('data_type', 'video'),
                            "fps": 1.0,
                            "max_frames": script_args.max_frames,
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(example.get('problem_type', "free-form"), "")
                        }
                    ]
                }
            ]
        }
        return msg

    dataset = dataset.map(make_conversation_image_and_video)
    # Initialize trainer class (keeps original selection)
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # pass masker to trainer if mask not added to dataset (so trainer can do dynamic masking in __getitem__)
    # create masker with given configuration for runtime use
    runtime_masker = None
    runtime_mask_cfg = MaskConfig(
        photometric_prob=script_args.photometric_prob,
        weather_prob=script_args.weather_prob,
        occlusion_prob=script_args.occlusion_prob,
        shake_prob=script_args.shake_prob,
        occlusion_mask_ratio=script_args.occlusion_mask_ratio,
        occlusion_block_mean=script_args.occlusion_block_mean,
        occlusion_block_std=script_args.occlusion_block_std,
        mask_value=script_args.mask_value,
        enable_temporal_mask=script_args.enable_temporal_mask and script_args.temporal,
        frame_mask_ratio=script_args.frame_mask_ratio,
        temporal_mode=script_args.temporal_mode,
        temporal_segment_len=script_args.temporal_segment_len,
        keep_k_frames=script_args.keep_k_frames,
        lighting_type=script_args.lighting_type,
        lighting_intensity=script_args.lighting_intensity,
        weather_type=script_args.weather_type,
        weather_particle_density=script_args.weather_particle_density,
        weather_particle_size=script_args.weather_particle_size,
        weather_speed=script_args.weather_speed,
        weather_effect_intensity=script_args.weather_effect_intensity,
        shake_intensity=script_args.shake_intensity,
        zoom_range=(script_args.zoom_min, script_args.zoom_max),
        rotation_range=(script_args.rotation_min, script_args.rotation_max),
        smoothness=script_args.smoothness,
        seed=42,
    )
    runtime_masker = VideoMasker(runtime_mask_cfg)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Attach masker to trainer for downstream use (trainer implementation must use it)
    trainer.video_masker = runtime_masker  # trainer code can check for this attribute and use it in __getitem__ or collate

    # Provide helper on trainer instance to compute KL given two logits tensors (user/trainer can call this)
    # def trainer_compute_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
    #     return compute_kl_from_logits(logits_p, logits_q, token_mask)

    # trainer.compute_kl_from_logits = trainer_compute_kl

    # Expose kl params for reward (so if reward funcs are invoked with kwargs, they can read these defaults)
    trainer.kl_alpha = script_args.kl_alpha

    # Start / resume training
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    trainer.save_memory(os.path.join(training_args.output_dir, "grpo_memory.json"))
    trainer.log_times()
    # Save and push to hub
    #trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
