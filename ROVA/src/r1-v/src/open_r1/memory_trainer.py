# GRPO trainer with memory management
# Adds memory management functionality on top of grpo_trainer.py

import os
import copy
import random
import numpy as np
from typing import Optional, Dict, Any, List

import torch
from transformers import Trainer
from qwen_vl_utils import process_vision_info

from memory_manager import MemoryManager, SufficiencyChecker, InsufficientSample
from video_mask import VideoMasker

from trainer.grpo_trainer import Qwen2VLGRPOTrainer
data_dir = "/path/to/data/"

class Qwen2VLGRPOTrainerWithMemory(Qwen2VLGRPOTrainer):
    """
    Extended GRPO Trainer with Memory management:
    1. Check sufficiency of each sample with random mask first
    2. Store insufficient samples in memory
    3. Trigger consolidation when memory is full: check solvability, train if solvable, mark dangerous otherwise
    4. Remove samples that are already dangerous and still unsolvable
    """
    
    def __init__(
        self,
        *args,
        enable_sufficiency_check: bool = True,
        memory_file: str = "insufficient_samples_memory.json",
        max_memory_size: int = 100,  # Maximum memory capacity
        recheck_interval: int = 1000,
        sufficiency_check_ratio: float = 1.0,  # Ratio of samples to perform sufficiency check
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.enable_sufficiency_check = enable_sufficiency_check
        self.sufficiency_check_ratio = sufficiency_check_ratio
        
        # Initialize Memory Manager
        self.memory_manager = MemoryManager(
            memory_file=memory_file,
            max_memory_size=max_memory_size,
            recheck_interval=recheck_interval
        )
        
        # Initialize Sufficiency Checker
        if self.enable_sufficiency_check:
            self.sufficiency_checker = SufficiencyChecker(
                model=self.model,
                processing_class=self.processing_class,
                device=self.accelerator.device
            )
        
        # Store trainable samples extracted from memory
        self.trainable_from_memory: List[Dict[str, Any]] = []
        
        # Flag indicating whether memory consolidation is in progress
        self.is_cleaning_memory = False
        
        print(f"[Memory] Initialized with enable_sufficiency_check={enable_sufficiency_check}")
        print(f"[Memory] Max memory size: {max_memory_size}")
    
    def _generate_mask_seed(self) -> int:
        """Generate random mask seed"""
        return random.randint(0, 2**31 - 1)
    
    def _create_masked_inputs_with_seed(
        self, 
        example: Dict[str, Any],
        seed: int
    ) -> Optional[Dict]:
        """
        Create masked input using specified seed
        """
        if self.video_masker is None:
            return None
        
        try:
            # Prepare input data
            input_copy = copy.deepcopy(example['prompt'])
            input_copy = self.remove_none_from_data(input_copy)
            
            # Set paths
            if example['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = data_dir + "/data" + example['path'][1:]
            elif example['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = data_dir + "/data" + example['path'][1:]

            # Process vision info
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
            except Exception as e:
                print(f"process_vision_info error in masking: {e}")
                return None
            
            # Prepare prompt text
            from trl.data_utils import maybe_apply_chat_template
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"]]
            
            # Process inputs
            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            
            # Temporarily set video_masker seed
            original_seed = self.video_masker.cfg.seed
            self.video_masker.cfg.seed = seed
            self.video_masker.rng = np.random.RandomState(seed)
            
            # Apply mask
            masked_inputs = self._apply_video_mask_to_prompt_inputs(prompt_inputs)
            
            # Restore original seed
            self.video_masker.cfg.seed = original_seed
            if original_seed is not None:
                self.video_masker.rng = np.random.RandomState(original_seed)
            else:
                self.video_masker.rng = np.random
            
            return masked_inputs
            
        except Exception as e:
            print(f"Error creating masked inputs: {e}")
            return None
    
    def _trigger_memory_cleanup(self):
        """
        Trigger memory consolidation process:
        1. Get all samples from memory
        2. Check if each sample can be answered with masked version
        3. Remove answerable samples and add to training queue
        4. Unanswerable: remove if dangerous, otherwise mark as dangerous
        """
        print(f"\n{'='*70}")
        print(f"[Memory] CLEANUP TRIGGERED - Memory is FULL ({len(self.memory_manager.memory)}/{self.memory_manager.max_memory_size})")
        print(f"{'='*70}")
        
        self.is_cleaning_memory = True
        
        # Get all samples that need checking
        samples_to_check = self.memory_manager.get_all_samples_for_cleanup()
        
        if not samples_to_check:
            print("[Memory] No samples to check (empty memory)")
            self.is_cleaning_memory = False
            return
        
        print(f"[Memory] Checking {len(samples_to_check)} samples...")
        
        # Create masked inputs for each sample
        masked_inputs_list = []
        valid_samples = []
        
        for sample in samples_to_check:
            # Rebuild example dict
            example = {
                'path': sample.path,
                'problem': sample.question,
                'data_type': sample.data_type,
                'problem_type': sample.problem_type,
                'solution': sample.solution,
                'prompt': [
                    {"role": "user", "content": sample.question}
                ]
            }
            
            # Create masked inputs using the same seed
            masked_inputs = self._create_masked_inputs_with_seed(example, sample.mask_seed)
            
            if masked_inputs is not None:
                masked_inputs_list.append(masked_inputs)
                valid_samples.append(sample)
            else:
                print(f"[Memory] Failed to create masked inputs for {sample.video_id}, skipping")
        
        if not valid_samples:
            print("[Memory] No valid samples to check")
            self.is_cleaning_memory = False
            return
        
        # Batch sufficiency check
        solvable, unsolvable = self.sufficiency_checker.batch_check_sufficiency(
            valid_samples,
            masked_inputs_list
        )
        
        # Process results
        trainable_samples = self.memory_manager.process_cleanup_results(
            solvable,
            unsolvable,
            self.state.global_step
        )
        
        # Convert trainable samples to training format and add to queue
        for sample in trainable_samples:
            example = {
                'path': sample.path,
                'problem': sample.question,
                'data_type': sample.data_type,
                'problem_type': sample.problem_type,
                'solution': sample.solution,
                'prompt': [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": sample.data_type,
                                "fps": 1
                            },
                            {
                                "type": "text",
                                "text": sample.question
                            }
                        ]
                    }
                ],
                '_mask_seed': sample.mask_seed,
                '_from_memory_cleanup': True
            }
            self.trainable_from_memory.append(example)
        
        print(f"[Memory] Cleanup complete: {len(trainable_samples)} samples ready for training")
        print(f"[Memory] Current memory size: {len(self.memory_manager.memory)}/{self.memory_manager.max_memory_size}")
        print(f"{'='*70}\n")
        
        self.memory_manager.save_memory()
        
        self.is_cleaning_memory = False
    
    def _check_single_sample_sufficiency(self, example: Dict[str, Any]) -> bool:
        """
        Perform sufficiency check on a single sample
        
        Returns:
            True if sufficient (continue training), False if insufficient (add to memory)
        """
        # Perform sufficiency check based on probability
        if random.random() > self.sufficiency_check_ratio:
            return True

        mask_seed = self._generate_mask_seed()
        masked_inputs = self._create_masked_inputs_with_seed(example, mask_seed)
        
        if masked_inputs is None:
            # Cannot create mask, train directly
            return True
        
        # Check sufficiency
        question = example.get('problem', '')
        is_sufficient = self.sufficiency_checker.check_sufficiency(
            masked_inputs, 
            question
        )
        
        if is_sufficient:
            # Sufficient to answer, continue training
            return True
        else:
            # Insufficient to answer, try adding to memory
            video_id = example.get('path', '') or f"sample_{random.randint(0, 1000000)}"
            
            success = self.memory_manager.add_sample(
                video_id=video_id,
                question=question,
                mask_seed=mask_seed,
                data_type=example.get('data_type', 'image'),
                path=example.get('path', ''),
                problem_type=example.get('problem_type', 'free-form'),
                solution=example.get('solution', ''),
                current_step=self.state.global_step
            )
            
            if not success:
                print(f"[Memory] Failed to add sample (memory full), will trigger cleanup")
            
            return False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add sufficiency check and memory management
        """
        # Check if memory consolidation needs to be triggered
        if self.memory_manager.is_full() and not self.is_cleaning_memory:
            self._trigger_memory_cleanup()
        
        # Prioritize trainable samples from memory if available
        if self.trainable_from_memory:
            num_from_memory = min(len(self.trainable_from_memory), len(inputs))
            memory_samples = self.trainable_from_memory[:num_from_memory]
            self.trainable_from_memory = self.trainable_from_memory[num_from_memory:]
            
            print(f"[Memory] Using {len(memory_samples)} samples from cleanup queue "
                  f"(remaining in queue: {len(self.trainable_from_memory)})")
            
            # Replace part of current batch with memory samples
            inputs[:num_from_memory] = memory_samples
        
        # Perform sufficiency check on non-memory samples
        sufficient_inputs = []
        
        for example in inputs:
            # Skip samples from memory
            if example.get('_from_memory_cleanup', False):
                sufficient_inputs.append(example)
                continue
            
            # Check sufficiency
            if self.enable_sufficiency_check:
                is_sufficient = self._check_single_sample_sufficiency(example)
                if is_sufficient:
                    sufficient_inputs.append(example)
            else:
                sufficient_inputs.append(example)
        
        if len(sufficient_inputs) == 0:
            print("[Memory] All samples insufficient, skipping this batch")
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        if len(sufficient_inputs) < len(inputs):
            print(f"[Memory] Filtered batch: {len(sufficient_inputs)}/{len(inputs)} samples sufficient")
            inputs = sufficient_inputs

        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        except Exception as e:
            print(f"[Memory] Error in compute_loss: {e}")
            return torch.tensor(0.001, device=self.accelerator.device, requires_grad=True)
    
    def train(self, *args, **kwargs):
        """Override train method to save memory at the end of training"""
        try:
            result = super().train(*args, **kwargs)
        finally:
            self.memory_manager.save_memory()

            stats = self.memory_manager.get_statistics()
            print(f"\n{'='*70}")
            print("[Memory] Final Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print(f"{'='*70}\n")
        
        return result
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save memory together with model"""
        super().save_model(output_dir, _internal_call)
        
        if output_dir and not _internal_call:
            memory_path = os.path.join(output_dir, "memory.json")
            original_file = self.memory_manager.memory_file
            self.memory_manager.memory_file = memory_path
            self.memory_manager.save_memory()
            self.memory_manager.memory_file = original_file
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Extend log method to add memory-related metrics"""
        stats = self.memory_manager.get_statistics()
        memory_logs = {
            'memory/size': stats['current_memory_size'],
            'memory/normal_samples': stats['normal_samples'],
            'memory/dangerous_samples': stats['dangerous_samples'],
            'memory/is_full': float(stats['is_full']),
            'memory/trainable_queue_size': len(self.trainable_from_memory),
        }
        logs.update(memory_logs)
        
        super().log(logs, start_time)