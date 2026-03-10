# memory_manager.py
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import torch
from collections import defaultdict

@dataclass
class InsufficientSample:
    """Store information about samples insufficient for answering"""
    video_id: str
    question: str
    mask_seed: int
    data_type: str  # 'image' or 'video'
    path: str
    problem_type: str
    solution: str
    timestamp: str
    check_count: int = 0  # Number of times checked
    last_check_step: int = 0  # Last training step when checked
    is_dangerous: bool = False  # Whether marked as dangerous
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class MemoryManager:
    """Memory system for managing samples insufficient for answering"""
    
    def __init__(
        self, 
        memory_file: str = "insufficient_samples_memory.json",
        max_memory_size: int = 100,  # Maximum memory capacity
        recheck_interval: int = 1000,  # Recheck interval (as fallback), mainly triggered by capacity
    ):
        self.memory_file = memory_file
        self.max_memory_size = max_memory_size
        self.recheck_interval = recheck_interval
        self.memory: Dict[str, InsufficientSample] = {}
        self.stats = defaultdict(int)
        
        self.load_memory()
        
        print(f"[Memory] Initialized with max_memory_size={max_memory_size}")
    
    def generate_sample_key(self, video_id: str, mask_seed: int) -> str:
        """Generate a unique sample key"""
        return f"{video_id}_{mask_seed}"
    
    def is_full(self) -> bool:
        """Check if memory is full"""
        return len(self.memory) >= self.max_memory_size
    
    def add_sample(
        self, 
        video_id: str,
        question: str,
        mask_seed: int,
        data_type: str,
        path: str,
        problem_type: str,
        solution: str,
        current_step: int = 0
    ) -> bool:
        """
        Add sample insufficient for answering to memory
        
        Returns:
            bool: Whether successfully added (returns False if memory is full)
        """
        key = self.generate_sample_key(video_id, mask_seed)
        
        # Check if already exists
        if key in self.memory:
            print(f"[Memory] Sample {key} already in memory, skipping")
            return True
        
        # Check capacity
        if self.is_full():
            print(f"[Memory] Memory is FULL ({len(self.memory)}/{self.max_memory_size}), cannot add new sample")
            return False
        
        # Add new sample
        sample = InsufficientSample(
            video_id=video_id,
            question=question,
            mask_seed=mask_seed,
            data_type=data_type,
            path=path,
            problem_type=problem_type,
            solution=solution,
            timestamp=datetime.now().isoformat(),
            check_count=0,
            last_check_step=current_step,
            is_dangerous=False
        )
        self.memory[key] = sample
        self.stats['total_added'] += 1
        
        print(f"[Memory] Added insufficient sample: {key} ({len(self.memory)}/{self.max_memory_size})")
        
        return True
    
    def remove_sample(self, video_id: str, mask_seed: int, reason: str = "sufficient"):
        """
        Remove sample from memory
        
        Args:
            reason: Removal reason - "sufficient"(now answerable) / "dangerous_unsolvable"(dangerous and still unsolvable)
        """
        key = self.generate_sample_key(video_id, mask_seed)
        if key in self.memory:
            sample = self.memory[key]
            del self.memory[key]
            
            if reason == "sufficient":
                self.stats['total_removed_sufficient'] += 1
                print(f"[Memory] Removed sample (now sufficient): {key}")
            elif reason == "dangerous_unsolvable":
                self.stats['total_removed_dangerous'] += 1
                print(f"[Memory] Removed DANGEROUS sample (still unsolvable): {key}")
            else:
                self.stats['total_removed_other'] += 1
                print(f"[Memory] Removed sample ({reason}): {key}")
    
    def mark_as_dangerous(self, video_id: str, mask_seed: int):
        """Mark sample as dangerous"""
        key = self.generate_sample_key(video_id, mask_seed)
        if key in self.memory:
            self.memory[key].is_dangerous = True
            self.stats['total_marked_dangerous'] += 1
            print(f"[Memory] Marked sample as DANGEROUS: {key}")
    
    def get_all_samples_for_cleanup(self) -> List[InsufficientSample]:
        """
        When memory is full, get all samples for consolidation check
        
        Returns:
            List of all samples in memory
        """
        return list(self.memory.values())
    
    def update_sample_check(self, video_id: str, mask_seed: int, current_step: int):
        """Update sample check information"""
        key = self.generate_sample_key(video_id, mask_seed)
        if key in self.memory:
            self.memory[key].check_count += 1
            self.memory[key].last_check_step = current_step
    
    def process_cleanup_results(
        self,
        solvable_samples: List[Tuple[str, int]],  # (video_id, mask_seed)
        unsolvable_samples: List[Tuple[str, int]],
        current_step: int
    ) -> List[InsufficientSample]:
        """
        Process memory consolidation results
        
        Args:
            solvable_samples: List of samples now solvable
            unsolvable_samples: List of samples still unsolvable
            current_step: Current training step
            
        Returns:
            List of samples available for training (solvable_samples)
        """
        trainable_samples = []
        
        # Process solvable samples - remove from memory, return for training
        for video_id, mask_seed in solvable_samples:
            key = self.generate_sample_key(video_id, mask_seed)
            if key in self.memory:
                sample = self.memory[key]
                trainable_samples.append(sample)
                self.remove_sample(video_id, mask_seed, reason="sufficient")
        
        # Process still unsolvable samples
        for video_id, mask_seed in unsolvable_samples:
            key = self.generate_sample_key(video_id, mask_seed)
            if key in self.memory:
                sample = self.memory[key]
                
                if sample.is_dangerous:
                    # Already dangerous and still unsolvable -> remove
                    self.remove_sample(video_id, mask_seed, reason="dangerous_unsolvable")
                else:
                    # First time unsolvable -> mark as dangerous, keep
                    self.mark_as_dangerous(video_id, mask_seed)
                    # Update check count
                    self.update_sample_check(video_id, mask_seed, current_step)
        
        print(f"[Memory] Cleanup results: {len(trainable_samples)} trainable, "
              f"{len(unsolvable_samples)} still unsolvable, "
              f"memory size now: {len(self.memory)}/{self.max_memory_size}")
        
        return trainable_samples
    
    def save_memory(self):
        """Save memory to file"""
        data = {
            'samples': {k: v.to_dict() for k, v in self.memory.items()},
            'stats': dict(self.stats),
            'config': {
                'max_memory_size': self.max_memory_size,
                'recheck_interval': self.recheck_interval
            },
            'last_save': datetime.now().isoformat()
        }
        
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[Memory] Saved {len(self.memory)} samples to {self.memory_file}")
    
    def load_memory(self):
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.memory = {
                    k: InsufficientSample.from_dict(v) 
                    for k, v in data.get('samples', {}).items()
                }
                self.stats = defaultdict(int, data.get('stats', {}))
                
                if 'config' in data:
                    cfg = data['config']
                    self.max_memory_size = cfg.get('max_memory_size', self.max_memory_size)
                    self.recheck_interval = cfg.get('recheck_interval', self.recheck_interval)
                
                print(f"[Memory] Loaded {len(self.memory)} samples from {self.memory_file}")

                if len(self.memory) > self.max_memory_size:
                    print(f"[Memory] WARNING: Loaded memory size ({len(self.memory)}) "
                          f"exceeds max_memory_size ({self.max_memory_size})")
            except Exception as e:
                print(f"[Memory] Failed to load memory file: {e}")
                self.memory = {}
        else:
            print(f"[Memory] No existing memory file found, starting fresh")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        dangerous_count = sum(1 for s in self.memory.values() if s.is_dangerous)
        normal_count = len(self.memory) - dangerous_count
        
        return {
            'current_memory_size': len(self.memory),
            'max_memory_size': self.max_memory_size,
            'is_full': self.is_full(),
            'normal_samples': normal_count,
            'dangerous_samples': dangerous_count,
            'total_added': self.stats['total_added'],
            'total_removed_sufficient': self.stats['total_removed_sufficient'],
            'total_removed_dangerous': self.stats['total_removed_dangerous'],
            'total_marked_dangerous': self.stats['total_marked_dangerous'],
            'samples_by_check_count': self._count_by_check_count(),
        }
    
    def _count_by_check_count(self) -> Dict[int, int]:
        """Get sample distribution by check count"""
        counts = defaultdict(int)
        for sample in self.memory.values():
            counts[sample.check_count] += 1
        return dict(counts)
    
    def clear_memory(self):
        """Clear memory"""
        self.memory.clear()
        print("[Memory] Cleared all samples")


class SufficiencyChecker:
    """Check if masked video is sufficient to answer the question"""
    
    def __init__(self, model, processing_class, device):
        self.model = model
        self.processing_class = processing_class
        self.device = device
        
        # Sufficiency check prompt template
        self.sufficiency_prompt = (
            "Based on the provided video/image, can you answer the following question with confidence? "
            "Please respond with ONLY 'YES' if you have enough information to answer accurately, "
            "or 'NO' if the visual information is insufficient.\n\n"
            "Question: {question}\n\n"
            "Response (YES/NO):"
        )
    
    def check_sufficiency(
        self, 
        masked_inputs: Dict,
        question: str,
        generation_config = None
    ) -> bool:
        """
        Check if masked video/image is sufficient to answer the question
        
        Returns:
            True if sufficient, False if insufficient
        """
        check_prompt = self.sufficiency_prompt.format(question=question)
        
        try:
            filtered_inputs = {k: v for k, v in masked_inputs.items() if k != "_mask_info"}
            with torch.no_grad():
                outputs = self.model.generate(
                    **filtered_inputs,
                    max_new_tokens=10,
                    do_sample=False, 
                    pad_token_id=self.processing_class.pad_token_id,
                )

            response = self.processing_class.batch_decode(
                outputs[:, masked_inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )[0].strip().upper()

            if 'YES' in response:
                return True
            elif 'NO' in response:
                return False
            else:
                print(f"[Sufficiency] Unclear response: {response}, treating as insufficient")
                return False
                
        except Exception as e:
            print(f"[Sufficiency] Check failed: {e}, treating as sufficient (conservative)")
            return True 
    
    def batch_check_sufficiency(
        self,
        samples: List[InsufficientSample],
        masked_inputs_list: List[Dict]
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Batch check sufficiency of memory samples
        
        Returns:
            (solvable_samples, unsolvable_samples): 
            Lists of solvable and unsolvable sample (video_id, mask_seed) tuples respectively
        """
        solvable = []
        unsolvable = []
        
        for sample, masked_inputs in zip(samples, masked_inputs_list):
            is_sufficient = self.check_sufficiency(masked_inputs, sample.question)
            
            if is_sufficient:
                solvable.append((sample.video_id, sample.mask_seed))
                status = "SOLVABLE" + (" [was dangerous]" if sample.is_dangerous else "")
            else:
                unsolvable.append((sample.video_id, sample.mask_seed))
                status = "UNSOLVABLE" + (" [already dangerous]" if sample.is_dangerous else " [mark dangerous]")
            
            print(f"[Cleanup] Sample {sample.video_id}_{sample.mask_seed}: {status}")
        
        return solvable, unsolvable