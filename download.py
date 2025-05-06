from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="EPFL-VILAB/4M-21_XL",
    local_dir="ckpt",
    local_dir_use_symlinks=False  # Copies files instead of symlinks
)
