"""Central configuration. Change DEVICE to 'cpu' to force CPU execution."""

import torch

# ----- Data -----
DATA_ROOT = "data/MedIMeta"
DATASET_ID = "bus"
TASK_A_NAME = "case category"
TASK_B_NAME = "malignancy"
NUM_CLASSES_A = 3
NUM_CLASSES_B = 1
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

# ----- Model -----
# ENCODER_CHANNELS = [1, 32, 64, 128, 256]
# HEAD_HIDDEN = 128
# DROPOUT_HEAD = 0.5
# DROPOUT_MID = 0.3

ENCODER_CHANNELS = [1, 16, 32, 64, 128]
HEAD_HIDDEN = 256
DROPOUT_HEAD = 0.2
DROPOUT_MID  = 0.1

# ----- Loss weights -----
ALPHA = 2.0   # weight for Head A (3-class)
BETA = 0.5    # weight for Head B (binary)

# ----- Training -----
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE_SCHEDULER = 4
PATIENCE_EARLY_STOP = 15

SEED = 42

# ----- Device -----
# Set to "cpu" or "cuda" or None for auto-detection.
DEVICE_OVERRIDE = None

# ----- Encoder selection -----
# "cnn"      -> original 4-block ConvNet (fast, works on CPU)
# "resnet18" -> pretrained ResNet18 adapted for grayscale input
ENCODER_TYPE = "cnn"

# Only used when ENCODER_TYPE == "resnet18"
PRETRAINED = True

def get_device() -> torch.device:
    if DEVICE_OVERRIDE is not None:
        return torch.device(DEVICE_OVERRIDE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")