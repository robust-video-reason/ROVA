# Memory-Managed GRPO Training

## Overview

This module extends the GRPO training pipeline with memory management to handle difficult samples adaptively.

### Core Mechanism

- **Sufficiency Check**: Before training, check if masked video input contains enough information to answer the question
- **Memory Buffer**: Store difficult samples that fail the sufficiency check for later re-evaluation
- **Adaptive Curriculum**: Periodically re-evaluate memory samples and reintroduce them when the model is ready

## File Structure

```
├── memory_manager.py              # Core memory management module
├── memory_trainer.py              # Extended Trainer class with memory
├── grpo_memory.py                 # Main training script with memory
├── video_mask.py                  # Video masking utilities
└── insufficient_samples_memory.json  # Memory persistence file (auto-generated)
```

## Usage

### Basic Usage

```bash
bash src/scripts/run_grpo_video_memory.sh
```

### Key Parameters

#### Memory Management Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_memory_size` | 100 | Maximum memory capacity |
| `recheck_interval` | 1000 | Recheck interval for memory samples |
| `sufficiency_check_ratio` | 1.0 | Ratio of samples to check for sufficiency |

#### Masking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_ratio` | 0.3 | Spatial mask ratio |
| `frame_mask_ratio` | 0.2 | Temporal frame drop ratio |

## Workflow

### 1. Sample Processing During Training

For each training sample:
1. Create masked video input using VideoMasker
2. Check sufficiency (can the question be answered with masked input?)
3. If sufficient -> continue training
4. If insufficient -> add to memory buffer

### 2. Memory Consolidation When Full

When memory reaches capacity:
1. Re-evaluate all stored samples with the current model
2. Samples now solvable -> extract for training
3. Samples still unsolvable -> mark as dangerous or remove

## Monitoring Metrics

- `memory_size`: Current number of samples in memory
- `memory_add_count`: Total samples added to memory
- `memory_train_count`: Samples moved from memory to training
- `memory_dangerous_count`: Samples marked as dangerous

## Best Practices

1. Start with default parameters and adjust based on monitoring
2. Monitor memory size - if it fills too quickly, the mask ratio may be too high
3. Use the memory persistence file to resume training

## Troubleshooting

### Memory fills up quickly
- Reduce mask ratio to make more samples answerable
- Increase max_memory_size

### Many dangerous samples
- These are samples that remain unsolvable even after re-evaluation
- Consider reducing perturbation intensity

### Slow training speed
- Reduce sufficiency_check_ratio to check fewer samples
- Reduce recheck_interval

## Technical Details

### Seed Mechanism
Each sample in memory stores a mask seed, ensuring reproducible masking when the sample is re-evaluated.

### Dangerous Marking
Samples that fail re-evaluation are marked as dangerous. If they fail again, they are removed from memory.

### Training Queue
Samples extracted from memory during consolidation are placed in a training queue and mixed into subsequent batches.
