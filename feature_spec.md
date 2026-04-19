# Feature Specification — ASL Recognition System

## 1. Feature Options

| Feature Type | File | Output Shape | Speed | Best For |
|---|---|---|---|---|
| Hand landmarks | `src/features/landmarks.py` | `(63,)` | ~15ms | MLP model, real-time |
| Raw pixels | `src/features/pixels.py` | `(3, 224, 224)` | ~5ms | CNN / MobileNet |
| Pixels + bg removed | `src/features/bg_remove.py` + pixels | `(3, 224, 224)` | ~500ms | Noisy background images |

## 2. Landmark Feature Schema

- **Extractor**: `LandmarkExtractor` (MediaPipe Hands)
- **Input**: Any image (file path / np.ndarray / PIL.Image)
- **Output**: `np.ndarray` dtype `float32` shape `(63,)`
- **Layout**: `[x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]`
- **Normalization**: Wrist (index 0) subtracted → scaled to `[-1, 1]`
- **None returned**: when no hand is detected in the image

### Landmark index map
| Index | Name |
|---|---|
| 0 | WRIST (origin after normalization) |
| 1–4 | THUMB (CMC, MCP, IP, TIP) |
| 5–8 | INDEX FINGER (MCP, PIP, DIP, TIP) |
| 9–12 | MIDDLE FINGER |
| 13–16 | RING FINGER |
| 17–20 | PINKY |

## 3. Pixel Feature Schema

- **Extractor**: `PixelExtractor` (torchvision transforms)
- **Input**: Any image (file path / np.ndarray / PIL.Image)
- **Output**: `torch.Tensor` dtype `float32` shape `(3, 224, 224)`
- **Normalization**: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Train mode augmentations** (from `params.yaml`):
  - `RandomRotation(±15°)`
  - `RandomHorizontalFlip`
  - `ColorJitter(brightness=0.3)`
- **Eval mode**: resize + normalize only (no augmentation)

## 4. Background Removal Schema

- **File**: `src/features/bg_remove.py`
- **Methods**: `rembg` (U2-Net, preferred) or `grabcut` (OpenCV, fallback)
- **Output**: `np.ndarray` shape `(H, W, 3)` with background replaced by white
- **Use**: Pass result into `PixelExtractor` after removal

## 5. Rules

- `src/features/` must NOT import from `src/models/`
- All config values (image size, augmentation params) read from `params.yaml`
- Feature logic versioned separately from model logic via DVC stage `features`