"""
Full preprocessing pipeline for PCOS ultrasound images.

Applies a configurable sequence of:
  resize → CLAHE → Gaussian/Anisotropic Diffusion → Z-score normalization

Also handles data augmentation (rotation, flip, scale) at training time.
"""

import cv2
import numpy as np

try:
    from medpy.filter.smoothing import anisotropic_diffusion as medpy_ad
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False


class Preprocessor:
    """Image preprocessing pipeline driven by a YAML config dict.

    Args:
        config: Preprocessing config dict (from configs/preprocessing/*.yaml).
        input_size: Target spatial size (e.g. 224 or 380). If None, read from config.
    """

    def __init__(self, config: dict, input_size: int = 224):
        self.config = config
        self.steps = config.get("steps", {})
        self.aug_config = config.get("augmentation", {})
        self.input_size = input_size

    def apply(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """Apply the full preprocessing pipeline to a single image.

        Args:
            image: Input image as HxWxC uint8 numpy array (BGR from cv2).
            augment: If True, apply random augmentation (training only).

        Returns:
            Preprocessed image as HxWxC float32 numpy array.
        """
        # Step 0: Resize (bicubic)
        if self.steps.get("resize", True):
            image = cv2.resize(
                image,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_CUBIC,
            )

        # Step 1: CLAHE
        clahe_cfg = self.steps.get("clahe", {})
        if clahe_cfg.get("enabled", False):
            image = self._apply_clahe(image, clahe_cfg)

        # Step 2a: Gaussian blur
        gauss_cfg = self.steps.get("gaussian", {})
        if gauss_cfg.get("enabled", False):
            image = self._apply_gaussian(image, gauss_cfg)

        # Step 2b: Anisotropic Diffusion
        ad_cfg = self.steps.get("anisotropic_diffusion", {})
        if ad_cfg.get("enabled", False):
            image = self._apply_anisotropic_diffusion(image, ad_cfg)

        # Convert to float32 for z-score
        image = image.astype(np.float32)

        # Step 3: Z-score normalization (per channel)
        zscore_cfg = self.steps.get("zscore_normalize", {})
        if zscore_cfg.get("enabled", False):
            image = self._apply_zscore(image)

        # Augmentation (training only)
        if augment:
            image = self._apply_augmentation(image)

        return image

    # ------------------------------------------------------------------
    # Individual preprocessing steps
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_clahe(image: np.ndarray, cfg: dict) -> np.ndarray:
        """Apply CLAHE to each channel independently.

        Args:
            image: HxWxC uint8 image.
            cfg: CLAHE config with clip_limit and tile_size.

        Returns:
            CLAHE-enhanced image.
        """
        clip_limit = cfg.get("clip_limit", 2.0)
        tile_size = tuple(cfg.get("tile_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

        if len(image.shape) == 2:
            # Grayscale
            return clahe.apply(image)

        # Apply CLAHE per channel
        channels = cv2.split(image)
        enhanced = [clahe.apply(ch) for ch in channels]
        return cv2.merge(enhanced)

    @staticmethod
    def _apply_gaussian(image: np.ndarray, cfg: dict) -> np.ndarray:
        """Apply Gaussian blur.

        Args:
            image: Input image.
            cfg: Config with sigma value.

        Returns:
            Blurred image.
        """
        sigma = cfg.get("sigma", 1.0)
        # Kernel size derived from sigma: must be odd
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    @staticmethod
    def _apply_anisotropic_diffusion(image: np.ndarray, cfg: dict) -> np.ndarray:
        """Apply anisotropic diffusion (Perona-Malik).

        Uses medpy if available, otherwise falls back to a manual implementation.

        Args:
            image: Input image (uint8 or float).
            cfg: Config with iterations, kappa, gamma.

        Returns:
            Diffusion-filtered image.
        """
        niter = cfg.get("iterations", 20)
        kappa = cfg.get("kappa", 30)
        gamma = cfg.get("gamma", 0.1)

        if HAS_MEDPY:
            # medpy expects float input
            img_float = image.astype(np.float64)
            if len(img_float.shape) == 3:
                # Apply per channel
                channels = []
                for c in range(img_float.shape[2]):
                    filtered = medpy_ad(
                        img_float[:, :, c],
                        niter=niter,
                        kappa=kappa,
                        gamma=gamma,
                        option=2,  # Perona-Malik option 2
                    )
                    channels.append(filtered)
                result = np.stack(channels, axis=2)
            else:
                result = medpy_ad(
                    img_float, niter=niter, kappa=kappa, gamma=gamma, option=2
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            # Manual Perona-Malik implementation
            return _perona_malik(image, niter, kappa, gamma)

    @staticmethod
    def _apply_zscore(image: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization.

        Args:
            image: HxWxC float32 image.

        Returns:
            Normalized image with zero mean and unit variance per channel.
        """
        if len(image.shape) == 2:
            mean = image.mean()
            std = image.std() + 1e-8
            return (image - mean) / std

        for c in range(image.shape[2]):
            mean = image[:, :, c].mean()
            std = image[:, :, c].std() + 1e-8
            image[:, :, c] = (image[:, :, c] - mean) / std
        return image

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a single image.

        Augmentations (all stochastic):
          - Random rotation ±rotation degrees
          - Random horizontal flip
          - Random scale ±scale fraction

        Args:
            image: HxWxC float32 image.

        Returns:
            Augmented image.
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Rotation
        rotation = self.aug_config.get("rotation", 0)
        if rotation > 0:
            angle = np.random.uniform(-rotation, rotation)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )

        # Horizontal flip
        if self.aug_config.get("horizontal_flip", False):
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)

        # Scale
        scale_range = self.aug_config.get("scale", 0)
        if scale_range > 0:
            scale_factor = 1.0 + np.random.uniform(-scale_range, scale_range)
            M = cv2.getRotationMatrix2D(center, 0, scale_factor)
            image = cv2.warpAffine(
                image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )

        return image


# ------------------------------------------------------------------
# Fallback: manual Perona-Malik anisotropic diffusion
# ------------------------------------------------------------------

def _perona_malik(
    image: np.ndarray, niter: int, kappa: float, gamma: float
) -> np.ndarray:
    """Manual Perona-Malik anisotropic diffusion.

    Args:
        image: Input image (uint8).
        niter: Number of iterations.
        kappa: Conduction coefficient (controls sensitivity to edges).
        gamma: Integration constant (0 < gamma <= 0.25 for stability).

    Returns:
        Filtered image as uint8.
    """
    img = image.astype(np.float64)

    if len(img.shape) == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_perona_malik_2d(img[:, :, c], niter, kappa, gamma))
        return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)
    else:
        return np.clip(_perona_malik_2d(img, niter, kappa, gamma), 0, 255).astype(
            np.uint8
        )


def _perona_malik_2d(
    img: np.ndarray, niter: int, kappa: float, gamma: float
) -> np.ndarray:
    """2D Perona-Malik diffusion on a single-channel image."""
    for _ in range(niter):
        # Compute gradients in 4 directions
        delta_n = np.roll(img, -1, axis=0) - img
        delta_s = np.roll(img, 1, axis=0) - img
        delta_e = np.roll(img, -1, axis=1) - img
        delta_w = np.roll(img, 1, axis=1) - img

        # Conduction coefficients (option 2: wide regions)
        cn = np.exp(-(delta_n / kappa) ** 2)
        cs = np.exp(-(delta_s / kappa) ** 2)
        ce = np.exp(-(delta_e / kappa) ** 2)
        cw = np.exp(-(delta_w / kappa) ** 2)

        # Update
        img = img + gamma * (cn * delta_n + cs * delta_s + ce * delta_e + cw * delta_w)

    return img
