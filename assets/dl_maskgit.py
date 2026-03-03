from huggingface_hub import hf_hub_download
from local_paths import MODELS_DIR

# The VQ-GAN
hf_hub_download(
    repo_id="FoundationVision/LlamaGen",
    filename="vq_ds16_c2i.pt",
    local_dir=MODELS_DIR / "maskgit",
)

# Imagenet weights
hf_hub_download(
    repo_id="llvictorll/Halton-Maskgit",
    filename="ImageNet_384_large.pth",
    local_dir=MODELS_DIR / "maskgit",
)
