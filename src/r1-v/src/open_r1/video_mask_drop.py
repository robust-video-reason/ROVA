# enhanced_video_mask.py
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
import torch
from PIL import Image
import math

ArrayLike = Union[np.ndarray, torch.Tensor, List[Image.Image]]

@dataclass
class MaskConfig:
    # spatial (pixel) mask
    pixel_mask_ratio: float = 0.3        # fraction of pixels to mask (0-1)
    pixel_mask_mode: str = "random_pixel" # "random_pixel" or "random_block"
    block_size: Optional[int] = 16       # block size for block-mode (pixels)
    per_frame_pixel_mask: bool = True    # whether spatial mask is sampled independently per frame
    mask_value: Union[int, float] = 0    # value to fill masked pixels with (0-255 or 0-1)
    
    # temporal (frame) mask
    frame_mask_ratio: float = 0.2        # fraction of frames to mask / drop
    temporal_mode: str = "random_drop"   # "random_drop" or "drop_segments" or "keep_k"
    temporal_segment_len: int = 4        # used when temporal_mode == "drop_segments"
    keep_k_frames: Optional[int] = None  # used when temporal_mode == "keep_k"
    
    # TOKEN-LEVEL masking (new for Qwen2.5-VL)
    token_mask_ratio: float = 0.3        # fraction of tokens to mask
    token_mask_mode: str = "random"      # "random", "block", or "structured"
    token_block_size: int = 16           # for block masking at token level
    preserve_cls_token: bool = True      # don't mask first token if it's CLS
    
    # other
    same_spatial_mask_for_all_channels: bool = True
    seed: Optional[int] = None


class VideoMasker:
    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed) if cfg.seed is not None else np.random

    # ========================================================================
    # ORIGINAL FRAME-LEVEL MASKING METHODS (unchanged)
    # ========================================================================
    
    def _to_numpy(self, frames: ArrayLike) -> np.ndarray:
        if isinstance(frames, torch.Tensor):
            arr = frames.detach().cpu().numpy()
            if arr.ndim == 4 and arr.shape[1] in (1,3,4) and arr.shape[1] <= 4 and arr.shape[1] < arr.shape[2]:
                arr = np.transpose(arr, (0,2,3,1))
            return arr
        if isinstance(frames, list):
            arrs = [np.array(im) for im in frames]
            return np.stack(arrs, axis=0)
        if isinstance(frames, np.ndarray):
            arr = frames
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.ndim == 4:
                if arr.shape[1] in (1,3,4) and arr.shape[1] < arr.shape[2]:
                    arr = np.transpose(arr, (0,2,3,1))
            return arr
        raise ValueError("Unsupported frames type: must be torch.Tensor, numpy.ndarray, or list[PIL.Image]")

    def _to_output_type(self, arr: np.ndarray, example_input: ArrayLike) -> ArrayLike:
        if isinstance(example_input, torch.Tensor):
            return torch.from_numpy(arr)
        if isinstance(example_input, list):
            return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        return arr

    def _random_pixel_mask(self, H:int, W:int, ratio:float) -> np.ndarray:
        total = H*W
        k = int(math.ceil(total * ratio))
        flat_idx = self.rng.choice(total, size=k, replace=False)
        mask = np.ones((H*W,), dtype=np.uint8)
        mask[flat_idx] = 0
        mask = mask.reshape(H,W)
        return mask

    def _random_block_mask(self, H:int, W:int, block_size:int, ratio:float) -> np.ndarray:
        mask = np.ones((H,W), dtype=np.uint8)
        total = H*W
        target_masked = int(total * ratio)
        masked = 0
        tries = 0
        max_tries = 10000
        while masked < target_masked and tries < max_tries:
            h0 = self.rng.randint(0, max(1, H - block_size + 1))
            w0 = self.rng.randint(0, max(1, W - block_size + 1))
            h1 = min(H, h0 + block_size)
            w1 = min(W, w0 + block_size)
            slice_area = (h1 - h0) * (w1 - w0)
            new_masked = np.sum(mask[h0:h1, w0:w1] == 1)
            if new_masked > 0:
                mask[h0:h1, w0:w1] = 0
                masked = total - np.sum(mask)
            tries += 1
        return mask

    def apply_pixel_mask_to_frame(self, frame: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = frame[..., None]
        H, W = mask_2d.shape
        if frame.shape[0] != H or frame.shape[1] != W:
            raise ValueError(f"Frame shape {frame.shape[:2]} and mask shape {mask_2d.shape} mismatch")
        
        out = frame.copy()
        
        if self.cfg.same_spatial_mask_for_all_channels:
            for c in range(frame.shape[2]):
                out[:, :, c][mask_2d == 0] = self.cfg.mask_value
        else:
            mask_3d = np.repeat(mask_2d[:, :, np.newaxis], frame.shape[2], axis=2)
            out[mask_3d == 0] = self.cfg.mask_value
        
        return out

    def sample_spatial_mask(self, H:int, W:int) -> np.ndarray:
        if self.cfg.pixel_mask_mode == "random_pixel":
            return self._random_pixel_mask(H, W, self.cfg.pixel_mask_ratio)
        elif self.cfg.pixel_mask_mode == "random_block":
            return self._random_block_mask(H, W, self.cfg.block_size or 16, self.cfg.pixel_mask_ratio)
        else:
            raise ValueError(f"Unknown pixel_mask_mode: {self.cfg.pixel_mask_mode}")

    def sample_temporal_mask(self, T:int) -> np.ndarray:
        if self.cfg.temporal_mode == "random_drop":
            k = int(math.floor(T * (1 - self.cfg.frame_mask_ratio)))
            keep_idx = self.rng.choice(T, size=max(1,k), replace=False)
            mask = np.zeros((T,), dtype=np.uint8)
            mask[keep_idx] = 1
            return mask
        elif self.cfg.temporal_mode == "drop_segments":
            mask = np.ones((T,), dtype=np.uint8)
            n_drop = int(math.floor(T * self.cfg.frame_mask_ratio))
            seg_len = max(1, self.cfg.temporal_segment_len)
            tries = 0
            dropped = 0
            max_tries = 10000
            while dropped < n_drop and tries < max_tries:
                start = self.rng.randint(0, T)
                end = min(T, start + seg_len)
                new_dropped = np.sum(mask[start:end] == 1)
                if new_dropped > 0:
                    mask[start:end] = 0
                    dropped = np.sum(mask == 0)
                tries += 1
            return mask
        elif self.cfg.temporal_mode == "keep_k":
            if self.cfg.keep_k_frames is None:
                raise ValueError("keep_k_frames must be set for temporal_mode='keep_k'")
            k = max(1, min(T, int(self.cfg.keep_k_frames)))
            keep_idx = self.rng.choice(T, size=k, replace=False)
            mask = np.zeros((T,), dtype=np.uint8)
            mask[keep_idx] = 1
            return mask
        else:
            raise ValueError(f"Unknown temporal_mode: {self.cfg.temporal_mode}")

    def mask_video(self, frames: ArrayLike, return_masks: bool = True) -> Tuple[ArrayLike, Dict[str, np.ndarray]]:
        """
        Main entrypoint for frame-level masking.

        Args:
            frames: (T,H,W,C) numpy / torch tensor / list[PIL.Image]
            return_masks: whether to return mask maps in dict

        Returns:
            masked_frames (same type as input), masks dict with keys:
                - 'spatial_masks': (T,H,W) binary mask (1 keep, 0 masked)
                - 'temporal_mask': (T,) binary mask (1 keep, 0 masked)
        """
        orig_type = type(frames)
        arr = self._to_numpy(frames)
        T, H, W, C = arr.shape

        temporal_keep = self.sample_temporal_mask(T)
        spatial_masks = np.ones((T, H, W), dtype=np.uint8)

        out = arr.copy()
        for t in range(T):
            if self.cfg.per_frame_pixel_mask:
                mask2d = self.sample_spatial_mask(H, W)
            else:
                if t == 0:
                    shared_mask2d = self.sample_spatial_mask(H, W)
                mask2d = shared_mask2d

            spatial_masks[t] = mask2d
            if temporal_keep[t] == 0:
                out[t] = np.ones_like(out[t]) * self.cfg.mask_value
                spatial_masks[t] = np.zeros_like(mask2d)
            else:
                out[t] = self.apply_pixel_mask_to_frame(out[t], mask2d)

        masked_out = self._to_output_type(out, frames)
        masks = {"spatial_masks": spatial_masks, "temporal_mask": temporal_keep}
        return masked_out, masks

    # ========================================================================
    # NEW: TOKEN-LEVEL MASKING METHODS (for Qwen2.5-VL patch tokens)
    # ========================================================================
    
    def mask_tokens(
        self, 
        tokens: torch.Tensor, 
        return_masks: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Mask patch tokens from Qwen2.5-VL processor.
        
        Args:
            tokens: torch.Tensor of shape (num_tokens, feature_dim) or (batch, num_tokens, feature_dim)
                   e.g., (1564, 1176) for Qwen2.5-VL
            return_masks: whether to return mask information
            
        Returns:
            masked_tokens: same shape as input with masked tokens set to mask_value
            masks: dict with key 'token_mask' (num_tokens,) binary array (1=keep, 0=masked)
        """
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"tokens must be torch.Tensor, got {type(tokens)}")
        
        original_shape = tokens.shape
        original_device = tokens.device
        original_dtype = tokens.dtype
        
        # Handle both (N, D) and (B, N, D) shapes
        if tokens.dim() == 2:
            # (num_tokens, feature_dim)
            num_tokens, feature_dim = tokens.shape
            has_batch = False
        elif tokens.dim() == 3:
            # (batch, num_tokens, feature_dim)
            batch_size, num_tokens, feature_dim = tokens.shape
            has_batch = True
            # Process each batch item separately
            masked_batch = []
            all_masks = []
            for b in range(batch_size):
                masked_item, mask_dict = self.mask_tokens(
                    tokens[b], 
                    return_masks=return_masks
                )
                masked_batch.append(masked_item)
                if return_masks:
                    all_masks.append(mask_dict)
            
            masked_tokens = torch.stack(masked_batch, dim=0)
            if return_masks:
                masks = {"token_masks_batch": all_masks}
            else:
                masks = {}
            return masked_tokens, masks
        else:
            raise ValueError(f"Unexpected tokens shape: {tokens.shape}. Expected 2D or 3D tensor.")
        
        # Generate token mask
        token_mask = self._sample_token_mask(num_tokens)
        
        # Apply masking
        masked_tokens = tokens.clone()
        mask_indices = np.where(token_mask == 0)[0]
        
        if len(mask_indices) > 0:
            # Set masked tokens to mask_value
            masked_tokens[mask_indices] = self.cfg.mask_value
        
        masks = {"token_mask": token_mask} if return_masks else {}
        
        return masked_tokens, masks
    
    def _sample_token_mask(self, num_tokens: int) -> np.ndarray:
        """
        Sample which tokens to mask.
        
        Args:
            num_tokens: total number of tokens
            
        Returns:
            mask: (num_tokens,) binary array where 1=keep, 0=mask
        """
        start_idx = 1 if self.cfg.preserve_cls_token else 0
        maskable_tokens = num_tokens - start_idx
        
        if self.cfg.token_mask_mode == "random":
            # Random token masking
            num_to_mask = int(maskable_tokens * self.cfg.token_mask_ratio)
            mask_indices = self.rng.choice(
                maskable_tokens, 
                size=num_to_mask, 
                replace=False
            ) + start_idx
            
            mask = np.ones(num_tokens, dtype=np.uint8)
            mask[mask_indices] = 0
            
        elif self.cfg.token_mask_mode == "block":
            # Block masking (consecutive tokens)
            mask = np.ones(num_tokens, dtype=np.uint8)
            num_to_mask = int(maskable_tokens * self.cfg.token_mask_ratio)
            
            masked = 0
            tries = 0
            max_tries = 1000
            block_size = self.cfg.token_block_size
            
            while masked < num_to_mask and tries < max_tries:
                start = self.rng.randint(start_idx, num_tokens)
                end = min(num_tokens, start + block_size)
                
                # Count how many new tokens would be masked
                new_masked = np.sum(mask[start:end] == 1)
                if new_masked > 0:
                    mask[start:end] = 0
                    masked = num_tokens - start_idx - np.sum(mask[start_idx:])
                tries += 1
                
        elif self.cfg.token_mask_mode == "structured":
            # Structured masking (e.g., every N-th token)
            mask = np.ones(num_tokens, dtype=np.uint8)
            stride = int(1.0 / self.cfg.token_mask_ratio)
            mask[start_idx::stride] = 0
            
        else:
            raise ValueError(f"Unknown token_mask_mode: {self.cfg.token_mask_mode}")
        
        # Ensure CLS token is not masked if preserve_cls_token=True
        if self.cfg.preserve_cls_token and num_tokens > 0:
            mask[0] = 1
        
        return mask
    
    # ========================================================================
    # UNIFIED INTERFACE (auto-detect input type)
    # ========================================================================
    
    def mask(
        self, 
        inputs: Union[ArrayLike, torch.Tensor], 
        return_masks: bool = True
    ) -> Tuple[Union[ArrayLike, torch.Tensor], Dict[str, np.ndarray]]:
        """
        Unified masking interface that auto-detects input type.
        
        Args:
            inputs: Either:
                   - Frame data: (T,H,W,C) numpy/torch/PIL list for frame-level masking
                   - Token data: (N, D) or (B, N, D) torch.Tensor for token-level masking
            return_masks: whether to return mask information
            
        Returns:
            masked_inputs: same type/shape as input
            masks: dict containing mask information
        """
        # Detect input type
        if isinstance(inputs, torch.Tensor):
            # Check if it's token-like (2D or 3D with reasonable dimensions)
            if inputs.dim() == 2:
                # (N, D) - likely tokens
                if inputs.shape[1] > 100:  # feature_dim typically > 100
                    return self.mask_tokens(inputs, return_masks)
            elif inputs.dim() == 3:
                # Could be (B, N, D) tokens or (T, H, W) frames
                if inputs.shape[2] > 100:  # likely (B, N, D) tokens
                    return self.mask_tokens(inputs, return_masks)
                # Otherwise fall through to frame masking
        
        # Default to frame-level masking
        return self.mask_video(inputs, return_masks)
    
    def make_pair_original_and_masked(
        self, 
        inputs: Union[ArrayLike, torch.Tensor]
    ) -> Tuple[Union[ArrayLike, torch.Tensor], Union[ArrayLike, torch.Tensor], Dict[str, np.ndarray]]:
        """
        Return (original, masked, masks) for both frame and token inputs.
        """
        if isinstance(inputs, torch.Tensor) and inputs.dim() in [2, 3] and inputs.shape[-1] > 100:
            # Token masking
            original = inputs.clone()
            masked, masks = self.mask_tokens(inputs, return_masks=True)
            return original, masked, masks
        else:
            # Frame masking
            orig_np = self._to_numpy(inputs)
            masked, masks = self.mask_video(orig_np, return_masks=True)
            return self._to_output_type(orig_np, inputs), masked, masks
