# import torch
# from huggingface_hub import hf_hub_download
# from PIL import Image
# from transformers import CLIPProcessor

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# torch.set_float32_matmul_precision("high")

# # Check if weights/model.torchscript exists, if not, download it from huggingface
# repo_id = "yaswanth169/deepfake-detection"
# filename = "model.torchscript"

# model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="weights")

# # Load checkpoint
# model = torch.jit.load(model_path, map_location=DEVICE)

# # Load preprocessing function
# preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
# batch_images = torch.stack(
#     [preprocess(images=image, return_tensors="pt")["pixel_values"][0] for image in pillow_images]
# )

# # Set model to evaluation mode
# model.eval()

# # Move model to the correct device and dtype
# model = model.to(DEVICE).to(DTYPE)

# # Move inputs to the correct device and dtype
# batch_images = batch_images.to(DEVICE).to(DTYPE)

# with torch.no_grad():
#     with torch.autocast(device_type="cuda", dtype=DTYPE):
#         # Forward pass
#         output = model(batch_images)

#         softmax_output = output.softmax(dim=1).cpu().numpy()

# for path, (p_real, p_fake) in zip(paths, softmax_output):
#     print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")


import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPProcessor

# ------------------ Device / DType ------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_CUDA else "cpu"

# GTX 1650 (Turing) supports fp16, NOT bf16
CUDA_DTYPE = torch.float16

torch.set_float32_matmul_precision("high")
if USE_CUDA:
    torch.backends.cudnn.benchmark = True

# ------------------ Download / Load ------------------
repo_id = "yaswanth169/deepfake-detection"
filename = "model.torchscript"
model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="weights")

# Load TorchScript to chosen device
model = torch.jit.load(model_path, map_location=DEVICE)
model.eval().to(DEVICE)

# Try to detect model's parameter dtype for CPU path
try:
    MODEL_EXPECTED_DTYPE = next(model.parameters()).dtype
except Exception:
    MODEL_EXPECTED_DTYPE = torch.float32  # fallback

# ------------------ Preprocess ------------------
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

batch_images = torch.stack(
    [preprocess(images=img, return_tensors="pt")["pixel_values"][0] for img in pillow_images]
).to(DEVICE)

# On CUDA: keep float32 inputs, autocast handles downcast
# On CPU: cast to model's expected dtype (often bf16 in scripted graphs)
if not USE_CUDA:
    if MODEL_EXPECTED_DTYPE == torch.bfloat16:
        batch_images = batch_images.to(torch.bfloat16)
    else:
        batch_images = batch_images.to(torch.float32)

# ------------------ Inference ------------------
def run_in_chunks(tensor, chunk_size=4):
    """Run in smaller batches to avoid CUDA OOM."""
    outs = []
    with torch.no_grad():
        for i in range(0, tensor.size(0), chunk_size):
            chunk = tensor[i:i+chunk_size]
            if USE_CUDA:
                try:
                    model.half()  # save VRAM if supported
                except Exception:
                    pass
                with torch.autocast(device_type="cuda", dtype=CUDA_DTYPE):
                    out = model(chunk)
            else:
                out = model(chunk)
            outs.append(out.float().softmax(dim=1).cpu())
    return torch.cat(outs, dim=0).numpy()

softmax_output = run_in_chunks(batch_images, chunk_size=4)

# ------------------ Results ------------------
for path, (p_real, p_fake) in zip(paths, softmax_output):
    print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")
