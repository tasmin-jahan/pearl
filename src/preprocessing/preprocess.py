"""
Full preprocessing pipeline for PCOS ultrasound images.

Applies a configurable sequence of:
  letterbox-resize → CLAHE → Gaussian / Anisotropic Diffusion (or SRAD) → Z-score normalization

Resize is aspect-preserving by default: the longer side is scaled to
``input_size`` (224 for all current models) and the shorter side is
letterboxed with reflected edge pixels. This preserves follicle shape
geometry that would otherwise be distorted by naive stretching. Set
``steps.padding: "none"`` in the preprocessing YAML to revert to the
legacy stretch behaviour (used by the v1/v2 ablation configs).

Two denoising options are supported for the diffusion step:
  - "anisotropic_diffusion": generic Perona-Malik AD (driven by gradient magnitude,
    assumes additive, signal-independent noise).
  - "srad": Speckle-Reducing Anisotropic Diffusion (driven by local coefficient
    of variation; correct for ultrasound's multiplicative, signal-dependent
    speckle). This is the recommended denoiser for ultrasound.

Also handles data augmentation (rotation, flip, scale, JPEG-style compression,
light blur) at training time. The JPEG-style and blur augmentations address
the heavy JPEG compression (~q40) present in the source dataset: they teach
the model to be invariant to residual codec artefacts without destroying the
already-den-and-CLAHE-preprocessed signal.
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
        # Padding strategy for the resize step. Options:
        #   "none"     — stretch to (input_size, input_size); legacy.
        #   "reflect"  — letterbox, pad with edge-replicated pixels (default;
        #                recommended for medical imaging).
        #   "constant" — letterbox, pad with the configured pad_value.
        self.padding = self.steps.get("padding", "reflect").lower()
        self.pad_value = int(self.steps.get("pad_value", 0))

        # Noise-robustness augmentations: applied on top of the preprocessed
        # image at training time to make the model invariant to the JPEG codec
        # noise already present in the source dataset (94.5% of images are at
        # estimated q <= 60; median q ~ 40). Both default to disabled; the
        # v3 ablation YAMLs enable them at low probability.
        jpeg_aug_cfg = self.aug_config.get("jpeg_compression", {}) or {}
        self.jpeg_aug_enabled = bool(jpeg_aug_cfg.get("enabled", False))
        self.jpeg_aug_prob = float(jpeg_aug_cfg.get("p", 0.3))
        self.jpeg_aug_q_low = int(jpeg_aug_cfg.get("q_low", 50))
        self.jpeg_aug_q_high = int(jpeg_aug_cfg.get("q_high", 95))

        blur_aug_cfg = self.aug_config.get("light_blur", {}) or {}
        self.blur_aug_enabled = bool(blur_aug_cfg.get("enabled", False))
        self.blur_aug_prob = float(blur_aug_cfg.get("p", 0.2))
        self.blur_aug_sigma_low = float(blur_aug_cfg.get("sigma_low", 0.1))
        self.blur_aug_sigma_high = float(blur_aug_cfg.get("sigma_high", 1.5))

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """Aspect-preserving resize + letterbox pad.

        Resize so the LONGER side equals ``self.input_size`` (preserving
        aspect ratio), then pad the shorter side to reach a square of
        ``(input_size, input_size)``. The pad is replicated edge pixels
        by default — this avoids the dark-border confound that zero-
        padding introduces, and matches the convention used by YOLO
        and most medical-imaging pipelines.

        Args:
            image: HxWxC uint8 image.

        Returns:
            (input_size, input_size, C) uint8 image.
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((self.input_size, self.input_size, 3),
                            dtype=image.dtype)
        # 1) Scale so the longer side = input_size
        scale = self.input_size / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if (new_h, new_w) != (h, w):
            image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_CUBIC,
            )
        # 2) Pad to square on the shorter side
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        # Centre the image: half the pad on each side
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if self.padding == "none":
            # Edge case: caller explicitly wants stretch even with
            # this code path (e.g. ablations). Force square.
            return cv2.resize(
                image, (self.input_size, self.input_size),
                interpolation=cv2.INTER_CUBIC,
            )
        if self.padding == "constant":
            value = self.pad_value
        else:  # "reflect" (default) — replicate edge pixels
            value = None  # signal to cv2.copyMakeBorder below
        # cv2.copyMakeBorder accepts a scalar value OR a 4-tuple of BGR
        # values. For "reflect" we use borderType=BORDER_REFLECT_101.
        border_type = (
            cv2.BORDER_CONSTANT if self.padding == "constant"
            else cv2.BORDER_REFLECT_101
        )
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=border_type,
            value=(
                (value,) * image.shape[2]
                if (value is not None and image.ndim == 3)
                else (value if value is not None else 0)
            ),
        )

    def apply(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """Apply the full preprocessing pipeline to a single image.

        Args:
            image: Input image as HxWxC uint8 numpy array (BGR from cv2).
            augment: If True, apply random augmentation (training only).

        Returns:
            Preprocessed image as HxWxC float32 numpy array.
        """
        # Step 0: Resize (letterbox by default; set steps.padding: 'none'
        # to revert to the legacy stretch behaviour).
        if self.steps.get("resize", True):
            if self.padding == "none":
                image = cv2.resize(
                    image,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                image = self._resize_with_padding(image)

        # Step 1: CLAHE
        clahe_cfg = self.steps.get("clahe", {})
        if clahe_cfg.get("enabled", False):
            image = self._apply_clahe(image, clahe_cfg)

        # Step 2a: Gaussian blur
        gauss_cfg = self.steps.get("gaussian", {})
        if gauss_cfg.get("enabled", False):
            image = self._apply_gaussian(image, gauss_cfg)

        # Step 2b: Anisotropic Diffusion (generic Perona-Malik OR SRAD)
        ad_cfg = self.steps.get("anisotropic_diffusion", {})
        if ad_cfg.get("enabled", False):
            method = ad_cfg.get("method", "srad")  # default to SRAD for ultrasound
            if method == "srad":
                image = self._apply_srad(image, ad_cfg)
            else:
                image = self._apply_anisotropic_diffusion(image, ad_cfg)

        # Step 2c: SRAD-only legacy config (preferred entrypoint in v3)
        srad_cfg = self.steps.get("srad", {})
        if srad_cfg.get("enabled", False):
            image = self._apply_srad(image, srad_cfg)

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
    def _apply_srad(image: np.ndarray, cfg: dict) -> np.ndarray:
        """Apply Speckle-Reducing Anisotropic Diffusion (SRAD).

        SRAD is the right diffusion family for ultrasound's multiplicative
        speckle noise: it drives the diffusion coefficient from the local
        coefficient-of-variation rather than raw gradient magnitude.

        Args:
            image: Input image (uint8 or float).
            cfg: Config with iterations, kappa, gamma.

        Returns:
            Diffusion-filtered image as uint8.
        """
        niter = cfg.get("iterations", 20)
        kappa = cfg.get("kappa", 30)
        gamma = cfg.get("gamma", 0.1)
        return _srad(image, niter, kappa, gamma)

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

    @staticmethod
    def apply_imagenet_stats(image: np.ndarray) -> np.ndarray:
        """Normalize using ImageNet mean/std (for pretrained backbones).

        Args:
            image: HxWxC float32 image in [0, 255] (or [0, 1]) range.

        Returns:
            Image normalized with ImageNet channel statistics.
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if image.max() > 1.5:
            image = image / 255.0
        if len(image.shape) == 2:
            # Convert grayscale to 3-channel by repeating
            image = np.stack([image] * 3, axis=-1)
        return (image - mean) / std

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a single image.

        Augmentations (all stochastic):
          - Random rotation ±rotation degrees
          - Random horizontal flip
          - Random scale ±scale fraction
          - JPEG-style compression artefact (with probability ``p``)
          - Light Gaussian blur (with probability ``p``)

        The two noise-robustness augmentations are applied LAST so the
        geometric augmentations (rotation/flip/scale) operate on the same
        content the loss function will see. Both are off by default and
        must be enabled explicitly under ``augmentation.jpeg_compression``
        and ``augmentation.light_blur`` in the YAML.

        Args:
            image: HxWxC float32 image (already preprocessed: SRAD/CLAHE/zscore).

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

        # JPEG-style compression artefact (noise-robustness augmentation).
        # The input is float32 (z-scored). We quantize to 8-bit, re-encode
        # at a random quality in [q_low, q_high], then return to float.
        # This simulates the family of codec artefacts the source dataset
        # already exhibits (q ~ 40), so the model becomes invariant to it.
        if self.jpeg_aug_enabled and np.random.random() < self.jpeg_aug_prob:
            image = self._apply_jpeg_augmentation(image)

        # Light Gaussian blur (noise-robustness augmentation).
        # Suppresses JPEG ringing (1-2 px radius) without erasing the
        # anatomical edge signal that survives compression (5-20 px radius).
        if self.blur_aug_enabled and np.random.random() < self.blur_aug_prob:
            image = self._apply_blur_augmentation(image)

        return image

    def _apply_jpeg_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression on a preprocessed float32 image.

        The flow is:
          float32 (any range) → clip+quantize to uint8 → cv2.imencode JPEG
          at random quality → cv2.imdecode → cast back to float32.

        This operates on the post-zscore values; the round-trip is intended
        to inject the same kind of codec artefacts the raw images have, not
        to be physically meaningful as compression of a normalized tensor.

        Args:
            image: HxWxC float32 image.

        Returns:
            Same-shape float32 image with simulated JPEG artefacts.
        """
        q = np.random.randint(self.jpeg_aug_q_low, self.jpeg_aug_q_high + 1)
        # Clip to a finite range; z-scored images are usually in [-3, 3],
        # but cv2.imencode expects 8-bit pixel values.
        clipped = np.clip(image, 0.0, 255.0).astype(np.uint8)
        ok, buf = cv2.imencode(
            ".jpg", clipped,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(q)],
        )
        if not ok:
            return image  # fail-safe: return unchanged
        decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return image
        if decoded.dtype != np.float32:
            decoded = decoded.astype(np.float32)
        return decoded

    def _apply_blur_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply light Gaussian blur to a preprocessed image.

        Args:
            image: HxWxC float32 image.

        Returns:
            Blurred image, same shape and dtype.
        """
        sigma = float(np.random.uniform(
            self.blur_aug_sigma_low, self.blur_aug_sigma_high,
        ))
        # Kernel size derived from sigma; must be odd and >= 3.
        ksize = max(3, int(np.ceil(sigma * 6)) | 1)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)


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


# ------------------------------------------------------------------
# SRAD: Speckle-Reducing Anisotropic Diffusion (Yu & Acton, 2002)
# ------------------------------------------------------------------

def _srad_2d(
    img: np.ndarray, niter: int, kappa: float, gamma: float
) -> np.ndarray:
    """2D SRAD on a single-channel image (float64, normalized to [0,1]).

    Unlike generic AD, the diffusion coefficient is driven by the local
    coefficient of variation (q), which is the right statistic for
    multiplicative, signal-dependent (speckle) noise.

    Args:
        img: 2D float64 image (already in [0,1] range for SRAD numerics).
        niter: Number of iterations.
        kappa: Speckle-scale parameter (controls edge sensitivity, default
            often ~10-30 for ultrasound).
        gamma: Integration constant in (0, 0.25] for stability.

    Returns:
        Filtered image, same shape/dtype as input.
    """
    for _ in range(niter):
        # Boundary handling: reflect by zero-padding conceptually via rolls.
        north = np.roll(img, -1, axis=0)
        south = np.roll(img, 1, axis=0)
        east = np.roll(img, -1, axis=1)
        west = np.roll(img, 1, axis=1)

        # Instantaneous coefficient of variation (ICOV)
        eps = 1e-8
        num = (north + south + east + west - 4.0 * img) ** 2
        den = (north + south + east + west + 4.0 * img * img + eps)
        q2 = num / den
        # ICOV squared is the SRAD diffusion-coefficient input
        c = np.exp(-(q2 - q2.mean()) / (kappa * (q2.std() + eps)))

        # Diffusion flux (4 directions)
        flux_n = c * (north - img)
        flux_s = c * (south - img)
        flux_e = c * (east - img)
        flux_w = c * (west - img)

        img = img + gamma * (flux_n + flux_s + flux_e + flux_w)

    return img


def _srad(
    image: np.ndarray, niter: int, kappa: float, gamma: float
) -> np.ndarray:
    """Run SRAD on an image (uint8 or float).

    Args:
        image: 2D or 3D image.
        niter, kappa, gamma: SRAD hyperparameters.

    Returns:
        Image as uint8 in original value range.
    """
    img = image.astype(np.float64)
    val_range = img.max() - img.min()
    if val_range < 1e-8:
        return image
    # SRAD assumes multiplicative noise: keep value scaling consistent
    # by working on normalized values internally.
    img_min = img.min()
    img = (img - img_min) / val_range

    if len(img.shape) == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_srad_2d(img[:, :, c], niter, kappa, gamma))
        result = np.stack(channels, axis=2)
    else:
        result = _srad_2d(img, niter, kappa, gamma)

    # Restore to original value range and dtype
    result = result * val_range + img_min
    return np.clip(result, 0, 255).astype(np.uint8)
