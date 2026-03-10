# enhanced_video_mask.py
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict

import math
import numpy as np
import torch
from PIL import Image

from lighting_enhancer import LightingEffectGenerator
from weather_enhancer import WeatherEffectGenerator
from process_camera import CameraShakeSimulator

ArrayLike = Union[np.ndarray, torch.Tensor, List[Image.Image]]


@dataclass
class MaskConfig:
    # -----------------------------
    #  Probability of each mask type (0~1, sum <=1, remainder means no augmentation)
    # -----------------------------
    photometric_prob: float = 0.25   # Photometric augmentation (dusk/night/overexposure/shadow)
    weather_prob: float = 0.25      # Weather effects (rain/snow/hail)
    occlusion_prob: float = 0.25    # Occlusion (block mask)
    shake_prob: float = 0.25        # Camera shake (simulated motion)

    # -----------------------------
    # Occlusion (block mask) configuration
    # -----------------------------
    occlusion_mask_ratio: float = 0.3     # Area ratio of occluded region
    occlusion_block_mean: int = 32        # Block size mean (pixels)
    occlusion_block_std: float = 12.0     # Block size standard deviation
    mask_value: Union[int, float] = 0     # Occlusion/temporal mask fill value (0~255 or 0~1)

    # -----------------------------
    # Temporal frame-level mask (toggleable)
    # -----------------------------
    enable_temporal_mask: bool = True
    frame_mask_ratio: float = 0.2         # Fraction of frames to drop
    temporal_mode: str = "random_drop"    # "random_drop" | "drop_segments" | "keep_k"
    temporal_segment_len: int = 4         # Consecutive segment length for drop_segments
    keep_k_frames: Optional[int] = None   # Number of frames to keep in keep_k mode

    # -----------------------------
    # Photometric augmentation (lighting_enhancer) configuration
    # -----------------------------
    # lighting_type can be "dusk", "night", "overexposure", "shadows" or "random"
    lighting_type: str = "random"
    lighting_intensity: float = 0.7       # 0~1, corresponds to the intensity parameter

    # -----------------------------
    # Weather effects (weather_enhancer) configuration
    # -----------------------------
    # weather_type: "light_rain", "heavy_rain", "storm", "snow", "hail" or "random"
    weather_type: str = "random"
    weather_particle_density: float = 0.5         # intensity
    weather_particle_size: int = 2                # Particle size
    weather_speed: int = 8                        # Particle speed
    weather_effect_intensity: float = 0.7         # weather_intensity

    # -----------------------------
    # Camera shake (process_camera) configuration
    # -----------------------------
    shake_intensity: float = 0.02
    zoom_range: Tuple[float, float] = (0.95, 1.05)
    rotation_range: Tuple[float, float] = (-2.0, 2.0)
    smoothness: float = 0.1

    # -----------------------------
    # Others
    # -----------------------------
    same_spatial_mask_for_all_channels: bool = True
    seed: Optional[int] = None


class VideoMasker:
    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed) if cfg.seed is not None else np.random
        self.seed = cfg.seed if cfg.seed is not None else 42

        self.lighting_gen = LightingEffectGenerator(
            lighting_type=self.cfg.lighting_type,
            intensity=self.cfg.lighting_intensity
        )

        self.weather_gen = WeatherEffectGenerator(
            weather_type=self.cfg.weather_type,
            intensity=self.cfg.weather_particle_density,
            particle_size=self.cfg.weather_particle_size,
            speed=self.cfg.weather_speed,
            weather_intensity=self.cfg.weather_effect_intensity
        )

        self.shake_sim = CameraShakeSimulator(
            shake_intensity=self.cfg.shake_intensity,
            zoom_range=self.cfg.zoom_range,
            rotation_range=self.cfg.rotation_range,
            smoothness=self.cfg.smoothness
        )

    def reset_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    # ========================================================================
    # Basic utilities: type conversion
    # ========================================================================
    def _to_numpy(self, frames: ArrayLike) -> np.ndarray:
        """
        Convert to (T, H, W, C) numpy.uint8 / float array
        """
        if isinstance(frames, torch.Tensor):
            arr = frames.detach().cpu().numpy()
            # Convert (T, C, H, W) -> (T, H, W, C)
            if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[1] < arr.shape[2]:
                arr = np.transpose(arr, (0, 2, 3, 1))
            return arr

        if isinstance(frames, list):
            arrs = [np.array(im) for im in frames]
            return np.stack(arrs, axis=0)  # (T, H, W, C)

        if isinstance(frames, np.ndarray):
            arr = frames
            if arr.ndim == 3:
                arr = arr[None, ...]  # (T=1, H, W, C) or (1, C, H, W)
            if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[1] < arr.shape[2]:
                # (T, C, H, W) -> (T, H, W, C)
                arr = np.transpose(arr, (0, 2, 3, 1))
            return arr

        raise ValueError("Unsupported frames type: must be torch.Tensor, numpy.ndarray, or list[PIL.Image]")

    def _to_output_type(self, arr: np.ndarray, example_input: ArrayLike) -> ArrayLike:
        """
        Convert numpy result back to the same type as input
        """
        if isinstance(example_input, torch.Tensor):
            return torch.from_numpy(arr)
        if isinstance(example_input, list):
            return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        return arr

    # ========================================================================
    # Temporal frame-level mask
    # ========================================================================
    def sample_temporal_mask(self, T: int) -> np.ndarray:
        """
        Return (T,) binary vector: 1=keep frame, 0=fill entire frame with mask_value
        """
        if not self.cfg.enable_temporal_mask:
            return np.ones((T,), dtype=np.uint8)

        if self.cfg.temporal_mode == "random_drop":
            k = int(math.floor(T * (1 - self.cfg.frame_mask_ratio)))
            k = max(1, k)
            keep_idx = self.rng.choice(T, size=k, replace=False)
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
                raise ValueError("keep_k_frames must be set when temporal_mode='keep_k'")
            k = max(1, min(T, int(self.cfg.keep_k_frames)))
            keep_idx = self.rng.choice(T, size=k, replace=False)
            mask = np.zeros((T,), dtype=np.uint8)
            mask[keep_idx] = 1
            return mask

        else:
            raise ValueError(f"Unknown temporal_mode: {self.cfg.temporal_mode}")

    # ========================================================================
    # Occlusion: block mask (block size ~ Normal distribution)
    # ========================================================================
    def _sample_occlusion_mask(self, H: int, W: int) -> np.ndarray:
        """
        Generate 2D occlusion mask based on occlusion_mask_ratio and Gaussian block size
        Return (H, W) binary mask: 1=visible, 0=occluded
        """
        mask = np.ones((H, W), dtype=np.uint8)
        total = H * W
        target_masked = int(total * self.cfg.occlusion_mask_ratio)
        masked = 0
        tries = 0
        max_tries = 10000

        while masked < target_masked and tries < max_tries:
            # Block size follows N(mean, std^2), clipped to a reasonable range
            size = int(self.rng.normal(loc=self.cfg.occlusion_block_mean,
                                       scale=self.cfg.occlusion_block_std or 1.0))
            size = max(4, size)            # Minimum size limit
            size = min(size, max(H, W))    # Should not exceed image dimensions

            h0 = self.rng.randint(0, max(1, H - size + 1))
            w0 = self.rng.randint(0, max(1, W - size + 1))
            h1 = min(H, h0 + size)
            w1 = min(W, w0 + size)

            new_masked = np.sum(mask[h0:h1, w0:w1] == 1)
            if new_masked > 0:
                mask[h0:h1, w0:w1] = 0
                masked = total - np.sum(mask)

            tries += 1

        return mask

    def apply_pixel_mask_to_frame(self, frame: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
        """
        Apply 2D mask to a single frame (for occlusion)
        """
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

    # ========================================================================
    # Photometric augmentation (based on LightingEffectGenerator)
    # ========================================================================
    def _apply_photometric(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply photometric augmentation to a single frame.
        Converts RGB to BGR + uint8, processes, then converts back.
        """
        orig_dtype = frame.dtype
        orig_max = float(frame.max()) if frame.size > 0 else 1.0
        frame_u8 = frame
        if frame_u8.dtype != np.uint8:
            if np.issubdtype(orig_dtype, np.floating):
                scale = 255.0 if orig_max <= 1.0 + 1e-6 else 1.0
                frame_u8 = np.clip(frame * scale, 0, 255).astype(np.uint8)
            else:
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)

        frame_bgr = frame_u8[..., ::-1]

        # ===== 2. Lighting effect (BGR uint8) =====
        lighting_type = self.cfg.lighting_type
        if lighting_type == "random":
            lighting_type = self.rng.choice(
                ["dusk", "night", "overexposure", "shadows"]
            )
        self.lighting_gen.lighting_type = lighting_type
        self.lighting_gen.intensity = self.cfg.lighting_intensity

        processed_bgr = self.lighting_gen.process_frame(frame_bgr)

        processed_rgb_u8 = processed_bgr[..., ::-1]

        if np.issubdtype(orig_dtype, np.floating):
            if orig_max <= 1.0 + 1e-6:
                out = processed_rgb_u8.astype(np.float32) / 255.0
            else:
                out = processed_rgb_u8.astype(orig_dtype)
        else:
            out = processed_rgb_u8.astype(orig_dtype)

        return out

    # ========================================================================
    # Weather effects (based on WeatherEffectGenerator)
    # ========================================================================
    def _apply_weather(self, frame: np.ndarray, particles) -> Tuple[np.ndarray, list]:
        """
        Call WeatherEffectGenerator to add weather effects and particles to a single frame.
        particles: particle list from the previous frame (for temporal consistency)
        """

        orig_dtype = frame.dtype
        orig_max = float(frame.max()) if frame.size > 0 else 1.0
        frame_u8 = frame
        if frame_u8.dtype != np.uint8:
            if np.issubdtype(orig_dtype, np.floating):
                # Most likely in [0,1] range, otherwise treat as [0,255] and clip
                scale = 255.0 if orig_max <= 1.0 + 1e-6 else 1.0
                frame_u8 = np.clip(frame * scale, 0, 255).astype(np.uint8)
            else:
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)

        # (If input is RGB, convert to BGR here, then convert back later)
        frame_bgr = frame_u8[..., ::-1]
        #frame_bgr = frame_u8

        weather_type = self.cfg.weather_type
        if weather_type == "random":
            weather_type = self.rng.choice(
                ["light_rain", "heavy_rain", "storm", "snow", "hail"]
            )

        self.weather_gen.weather_type = weather_type
        self.weather_gen.intensity = self.cfg.weather_particle_density
        self.weather_gen.particle_size = self.cfg.weather_particle_size
        self.weather_gen.speed = self.cfg.weather_speed
        self.weather_gen.weather_intensity = self.cfg.weather_effect_intensity

        if particles is None:
            particles = self.weather_gen.generate_particles(frame_bgr.shape)

        # Update and add new particles
        particles = self.weather_gen.update_particles(particles, frame_bgr.shape)
        new_particles = self.weather_gen.generate_particles(frame_bgr.shape)
        particles.extend(new_particles)

        # First apply overall weather lighting/tone, then draw particles
        weather_frame = self.weather_gen.apply_weather_effects(frame_bgr)
        frame_with_weather = self.weather_gen.draw_particles(weather_frame, particles)

        frame_with_weather = frame_with_weather[..., ::-1] # Convert back to RGB

        if np.issubdtype(orig_dtype, np.floating):
            if orig_max <= 1.0 + 1e-6:
                out = frame_with_weather.astype(np.float32) / 255.0
            else:
                out = frame_with_weather.astype(orig_dtype)
        else:
            out = frame_with_weather.astype(orig_dtype)

        return out, particles

    # ========================================================================
    # Camera shake effect (based on CameraShakeSimulator)
    # ========================================================================
    def _generate_shake_params(self, T: int, H: int, W: int):
        """
        Generate a smooth camera motion curve for the entire video
        """
        # Sync internal parameters to current config
        self.shake_sim.shake_intensity = self.cfg.shake_intensity
        self.shake_sim.zoom_range = self.cfg.zoom_range
        self.shake_sim.rotation_range = self.cfg.rotation_range
        self.shake_sim.smoothness = self.cfg.smoothness

        dx, dy, zoom, rotation = self.shake_sim.generate_camera_motion(
            num_frames=T, width=W, height=H
        )
        return dx, dy, zoom, rotation

    def _apply_shake(self, frame: np.ndarray,
                     dx: float, dy: float, zoom: float, rotation: float) -> np.ndarray:
        return self.shake_sim.apply_transform(frame, dx, dy, zoom, rotation)

    # ========================================================================
    # Main function: mask_video
    # ========================================================================
    def mask_video(self, frames: ArrayLike, return_masks: bool = True) -> Tuple[ArrayLike, Dict[str, np.ndarray]]:
        """
        Apply mask / augmentation to video. Input:
            frames: (T,H,W,C) numpy / torch tensor / list[PIL.Image]

        Strategy:
            1. Sample temporal mask (optional), fill entire frame with mask_value.
            2. Select one mask category by probability for the entire video:
               - photometric: lighting augmentation
               - weather: weather effects (with particles)
               - occlusion: block occlusion (Gaussian block size)
               - shake: camera shake (simulated transform)
            3. Apply the chosen category to all temporally unmasked frames.
               If all probabilities are 0, skip augmentation.

        Returns:
            masked_frames (same type as input),
            masks dict:
                - 'spatial_masks': (T,H,W) 1=visible, 0=occluded (all 1 for photometric/weather/shake)
                - 'temporal_mask': (T,) 1=kept frame, 0=entire frame masked
        """
        arr = self._to_numpy(frames)
        T, H, W, C = arr.shape

        temporal_mask = self.sample_temporal_mask(T)
        spatial_masks = np.ones((T, H, W), dtype=np.uint8)
        out = arr.copy()

        probs = np.array([
            self.cfg.photometric_prob,
            self.cfg.weather_prob,
            self.cfg.occlusion_prob,
            self.cfg.shake_prob
        ], dtype=np.float32)
        prob_sum = float(probs.sum())
        categories = ["photometric", "weather", "occlusion", "shake"]

        chosen_category = None
        if prob_sum > 0:
            probs = probs / prob_sum
            r = self.rng.rand()
            cum = 0.0
            for idx, cat in enumerate(categories):
                cum += probs[idx]
                if r < cum:
                    chosen_category = cat
                    break

        dx = dy = zoom = rot = None
        if chosen_category == "shake":
            dx, dy, zoom, rot = self._generate_shake_params(T, H, W)

        weather_particles = None  # Weather particles shared across frames

        for t in range(T):
            frame = out[t]

            # Temporal dimension: overwrite masked frames directly
            if temporal_mask[t] == 0:
                frame[...] = self.cfg.mask_value
                out[t] = frame
                spatial_masks[t] = 0
                continue

            # No category selected (all prob=0), skip augmentation
            if chosen_category is None:
                out[t] = frame
                spatial_masks[t] = 1
                continue

            # Process based on the unified chosen_category for the whole video
            if chosen_category == "photometric":
                out[t] = self._apply_photometric(frame)
                spatial_masks[t] = 1 

            elif chosen_category == "weather":
                out[t], weather_particles = self._apply_weather(frame, weather_particles)
                spatial_masks[t] = 1 

            elif chosen_category == "occlusion":
                mask_2d = self._sample_occlusion_mask(H, W)
                spatial_masks[t] = mask_2d
                out[t] = self.apply_pixel_mask_to_frame(frame, mask_2d)

            elif chosen_category == "shake":
                out[t] = self._apply_shake(frame, dx[t], dy[t], zoom[t], rot[t])
                spatial_masks[t] = 1

        masked_out = self._to_output_type(out, frames)
        masks = {"spatial_masks": spatial_masks, "temporal_mask": temporal_mask} if return_masks else {}
        #print( f"Applied video mask category: {chosen_category}" )
        return masked_out, masks

