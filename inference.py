# import torch
# from huggingface_hub import hf_hub_download
# from lightning.fabric import Fabric
# from PIL import Image

# from src.config import Config
# from src.model.dfdet import DeepfakeDetectionModel

# DEVICES = [0]

# torch.set_float32_matmul_precision("high")

# # Check if weights/model.ckpt exists, if not, download it from huggingface
# repo_id = "yaswanth169/deepfake-detection"
# filename = "model.ckpt"

# model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="weights")

# # Load checkpoint
# ckpt = torch.load(model_path, map_location="cpu")

# run_name = ckpt["hyper_parameters"]["run_name"]
# print(run_name)

# # Initialize model from config
# model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
# model.eval()

# # Load model state dict
# model.load_state_dict(ckpt["state_dict"])

# # Get preprocessing function
# preprocessing = model.get_preprocessing()

# # Load some images
# paths = [
#     "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/000.png",
#     "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/045.png",
#     "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/030.png",
#     "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/015.png",
#     "datasets/CDFv2/YouTube-real/00000/000.png",
#     "datasets/CDFv2/YouTube-real/00000/014.png",
#     "datasets/CDFv2/YouTube-real/00000/028.png",
#     "datasets/CDFv2/YouTube-real/00000/043.png",
#     "datasets/CDFv2/Celeb-real/id0_0000/045.png",
#     "datasets/CDFv2/Celeb-real/id0_0000/030.png",
#     "datasets/CDFv2/Celeb-real/id0_0000/015.png",
#     "datasets/CDFv2/Celeb-real/id0_0000/000.png",
# ]

# # To pillow images
# pillow_images = [Image.open(image) for image in paths]

# # To tensors
# batch_images = torch.stack([preprocessing(image) for image in pillow_images])

# precision = ckpt["hyper_parameters"]["precision"]
# fabric = Fabric(accelerator="cuda", devices=DEVICES, precision=precision)
# fabric.launch()
# model = fabric.setup_module(model)

# # perform inference
# with torch.no_grad():
#     # Move batch_images to the correct device and dtype
#     batch_images = batch_images.to(fabric.device).to(model.dtype)

#     # Forward pass
#     output = model(batch_images)

# # logits to probabilities
# softmax_output = output.logits_labels.softmax(dim=1).cpu().numpy()

# for path, (p_real, p_fake) in zip(paths, softmax_output):
#     print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")


# inference.py  — runs on CPU or CUDA automatically

import torch
from huggingface_hub import hf_hub_download
from lightning.fabric import Fabric
from PIL import Image

from src.config import Config
from src.model.dfdet import DeepfakeDetectionModel

# -------------------- Device / precision --------------------
HAS_CUDA = torch.cuda.is_available()
ACCELERATOR = "cuda" if HAS_CUDA else "cpu"
# Let the checkpoint decide precision on CUDA; force 32-bit on CPU
CPU_PRECISION = "32-true"  # Fabric-friendly: full float32 on CPU

torch.set_float32_matmul_precision("high")

# -------------------- Weights --------------------
repo_id = "yaswanth169/deepfake-detection"
filename = "model.ckpt"
model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="weights")

# -------------------- Load checkpoint & model --------------------
ckpt = torch.load(model_path, map_location="cpu")
run_name = ckpt["hyper_parameters"]["run_name"]
print(run_name)

# Build model from saved hyperparameters
model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
model.eval()
model.load_state_dict(ckpt["state_dict"])

# Preprocessing callable saved by the model
preprocessing = model.get_preprocessing()

# -------------------- Demo inputs --------------------
paths = [
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/000.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/045.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/030.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/015.png",
    "datasets/CDFv2/YouTube-real/00000/000.png",
    "datasets/CDFv2/YouTube-real/00000/014.png",
    "datasets/CDFv2/YouTube-real/00000/028.png",
    "datasets/CDFv2/YouTube-real/00000/043.png",
    "datasets/CDFv2/Celeb-real/id0_0000/045.png",
    "datasets/CDFv2/Celeb-real/id0_0000/030.png",
    "datasets/CDFv2/Celeb-real/id0_0000/015.png",
    "datasets/CDFv2/Celeb-real/id0_0000/000.png",
]
pillow_images = [Image.open(p).convert("RGB") for p in paths]
batch_images = torch.stack([preprocessing(img) for img in pillow_images])  # (N, C, H, W)

# -------------------- Fabric init (auto CPU/GPU) --------------------
ckpt_precision = ckpt["hyper_parameters"].get("precision", "bf16-mixed")
precision = ckpt_precision if HAS_CUDA else CPU_PRECISION

# If you have multiple GPUs and want a specific one, set devices=[0] or list of ids.
devices = 1 if not HAS_CUDA else [0]

fabric = Fabric(accelerator=ACCELERATOR, devices=devices, precision=precision)
fabric.launch()

model = fabric.setup_module(model)

# -------------------- Inference (with optional chunking) --------------------
def run_in_chunks(tensor: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
    """Run model in smaller chunks to reduce VRAM/RAM pressure."""
    outs = []
    with torch.no_grad():
        for i in range(0, tensor.size(0), chunk_size):
            chunk = tensor[i : i + chunk_size].to(fabric.device)
            # Make sure dtype matches model dtype
            chunk = chunk.to(model.dtype)
            out = model(chunk)  # returns object with .logits_labels
            outs.append(out.logits_labels.float().softmax(dim=1).cpu())
    return torch.cat(outs, dim=0)

with torch.no_grad():
    # Move big batch once (CPU users can keep it on CPU; CUDA copies inside chunk loop anyway)
    batch_images = batch_images  # tensors moved per-chunk above

    # Adjust chunk_size if you hit OOM on GPU/CPU
    probs = run_in_chunks(batch_images, chunk_size=4).numpy()

# -------------------- Results --------------------
for path, (p_real, p_fake) in zip(paths, probs):
    print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")
