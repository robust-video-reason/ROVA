import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from memory_trainer import Qwen2VLGRPOTrainerWithMemory
from video_mask import MaskConfig, VideoMasker

from grpo import (
    reward_funcs_registry, 
    SYSTEM_PROMPT, 
    accuracy_reward,
    format_reward,
    kl_consistency_reward,
    compute_kl_from_logits
)

@dataclass
class GRPOScriptArgumentsWithMemory(ScriptArguments):
    
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions"},
    )
    max_pixels: Optional[int] = field(default=524288)
    min_pixels: Optional[int] = field(default=3136)
    temporal: Optional[bool] = field(default=True)
    len_control: Optional[bool] = field(default=True)
    
    # Masking parameters
    mask_in_dataset_map: Optional[bool] = field(default=False)
    pixel_mask_ratio: Optional[float] = field(default=0.3)
    pixel_mask_mode: Optional[str] = field(default="random_pixel")
    block_size: Optional[int] = field(default=16)
    per_frame_pixel_mask: Optional[bool] = field(default=True)
    frame_mask_ratio: Optional[float] = field(default=0.2)
    temporal_mode: Optional[str] = field(default="random_drop")
    temporal_segment_len: Optional[int] = field(default=4)
    keep_k_frames: Optional[int] = field(default=None)
    kl_alpha: Optional[float] = field(default=1.0)
    kl_reward_name: Optional[str] = field(default="kl_consistency")
    
    # Memory management parameters
    enable_sufficiency_check: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to enable sufficiency check and memory management"}
    )
    memory_file: Optional[str] = field(
        default="insufficient_samples_memory.json",
        metadata={"help": "Memory file path"}
    )
    max_memory_size: Optional[int] = field(
        default=100,
        metadata={"help": "Maximum memory capacity, triggers consolidation when full"}
    )
    recheck_interval: Optional[int] = field(
        default=1000,
        metadata={"help": "Recheck interval (training steps) - mainly triggered by capacity"}
    )
    sufficiency_check_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "Ratio of samples for sufficiency check (0-1)"}
    )


def main(script_args, training_args, model_args):
    """Main training function"""
    
    print("\n" + "="*70)
    print("GRPO Training with Memory Management")
    print("="*70)
    
    # 1. Load reward functions
    print("\n[1/7] Loading reward functions...")
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print(f"  Loaded {len(reward_funcs)} reward functions: {script_args.reward_funcs}")
    
    # 2. Load dataset
    print("\n[2/7] Loading dataset...")
    print(f"  Dataset: {script_args.dataset_name}")
    
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    print(f"  Train split: {len(dataset[script_args.dataset_train_split])} samples")
    if script_args.dataset_test_split in dataset:
        print(f"  Test split: {len(dataset[script_args.dataset_test_split])} samples")
    
    # 3. Data processing - convert to conversation format
    print("\n[3/7] Processing dataset...")
    
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
                            "type": example.get('data_type', 'image'),
                            "fps": 1
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
    print("  Dataset processed successfully")
    
    # 4. Create VideoMasker
    print("\n[4/7] Initializing VideoMasker...")
    mask_cfg = MaskConfig(
        pixel_mask_ratio=script_args.pixel_mask_ratio,
        pixel_mask_mode=script_args.pixel_mask_mode,
        block_size=script_args.block_size,
        per_frame_pixel_mask=script_args.per_frame_pixel_mask,
        mask_value=0,
        frame_mask_ratio=script_args.frame_mask_ratio,
        temporal_mode=script_args.temporal_mode,
        temporal_segment_len=script_args.temporal_segment_len,
        keep_k_frames=script_args.keep_k_frames,
        seed=None  # Dynamic masking, no fixed seed
    )
    runtime_masker = VideoMasker(mask_cfg)
    print(f"  Pixel mask: {script_args.pixel_mask_ratio} ({script_args.pixel_mask_mode})")
    print(f"  Frame mask: {script_args.frame_mask_ratio} ({script_args.temporal_mode})")
    
    # 5. Initialize Trainer with Memory management
    print("\n[5/7] Initializing Trainer with Memory Management...")
    trainer = Qwen2VLGRPOTrainerWithMemory(
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
        # Memory management parameters
        enable_sufficiency_check=script_args.enable_sufficiency_check,
        memory_file=script_args.memory_file,
        max_memory_size=script_args.max_memory_size,
        recheck_interval=script_args.recheck_interval,
        sufficiency_check_ratio=script_args.sufficiency_check_ratio,
    )
    
    # Attach masker to trainer
    trainer.video_masker = runtime_masker
    print("  VideoMasker attached to trainer")

    try:
        def trainer_compute_kl(logits_p, logits_q, token_mask=None):
            return compute_kl_from_logits(logits_p, logits_q, token_mask)
        
        trainer.compute_kl_from_logits = trainer_compute_kl
        trainer.kl_alpha = script_args.kl_alpha
        print("  KL computation helper attached")
    except ImportError:
        print("  [Warning] KL computation helper not available")
    
    # 6. Print configuration summary
    print("\n[6/7] Configuration Summary:")
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Output dir: {training_args.output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print("\n  Memory Management:")
    print(f"    Enabled: {script_args.enable_sufficiency_check}")
    print(f"    Max size: {script_args.max_memory_size}")
    print(f"    Check ratio: {script_args.sufficiency_check_ratio}")
    print(f"    Memory file: {script_args.memory_file}")
    
    # 7. Start training
    print("\n[7/7] Starting Training...")
    print("="*70 + "\n")
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    
    # 8. Save and push to hub
    print("\n" + "="*70)
    print("Training completed! Saving model...")
    print("="*70)
    
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to: {training_args.output_dir}")
    
    if training_args.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        print("Model pushed to hub successfully!")
    
    # 9. Print final memory statistics
    print("\n" + "="*70)
    print("Final Memory Statistics:")
    print("="*70)
    stats = trainer.memory_manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    print("✅ All done!")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArgumentsWithMemory, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print("Starting GRPO Training with Memory Management System")
    
    main(script_args, training_args, model_args)