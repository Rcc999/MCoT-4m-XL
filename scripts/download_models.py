#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

MODELS = {
    "4m-xl-21-finetuned": {
        "gdrive_folder": "1xTY4k4zEJIOC6vo6o9qEA3Xh_dfulttf",
        "gdrive_url": "https://drive.google.com/drive/folders/1xTY4k4zEJIOC6vo6o9qEA3Xh_dfulttf?usp=sharing",
        "size_gb": 15.2,
        "description": "4M-XL-21 model fine-tuned on VQAv2"
    },
    "4m-xl-t2i": {
        "gdrive_file": "1CVHUH2kLJHLEYjeYJ_k2XcuYPmWXR3Sp",
        "gdrive_url": "https://drive.google.com/file/d/1CVHUH2kLJHLEYjeYJ_k2XcuYPmWXR3Sp/view?usp=sharing",
        "size_gb": 12.8,
        "description": "4M-XL Text-to-Image model"
    }
}

def check_rclone():
    try:
        result = subprocess.run(['rclone', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("rclone found")
            return True
    except FileNotFoundError:
        pass
    
    print("rclone not found")
    return False

def install_rclone():
    print("Installing rclone...")
    try:
        if sys.platform.startswith('linux'):
            subprocess.run(['curl', 'https://rclone.org/install.sh'], stdout=subprocess.PIPE)
            subprocess.run(['sudo', 'bash'], input=subprocess.run(['curl', 'https://rclone.org/install.sh'], capture_output=True).stdout)
        elif sys.platform == 'darwin':
            subprocess.run(['brew', 'install', 'rclone'])
        else:
            print("Automatic rclone installation not supported on this platform")
            print("Please install rclone manually: https://rclone.org/install/")
            return False
        return True
    except Exception as e:
        print(f"Failed to install rclone: {e}")
        return False

def setup_rclone_gdrive():
    print("\nSetting up rclone for Google Drive...")
    print("1. Run: rclone config")
    print("2. Choose 'n' for new remote")
    print("3. Name it 'gdrive'")
    print("4. Choose 'Google Drive' (option 15)")
    print("5. Leave client_id and client_secret blank")
    print("6. Choose 'y' for full access")
    print("7. Leave root_folder_id blank")
    print("8. Leave service_account_file blank") 
    print("9. Choose 'n' for advanced config")
    print("10. Choose 'y' for auto config")
    print("11. Follow browser authentication")
    print("12. Choose 'n' for team drive")
    print("13. Choose 'y' to confirm")
    print("14. Choose 'q' to quit config")
    
    input("\nPress Enter after completing rclone config...")

def download_with_rclone(model_name, model_info):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_dir = models_dir / model_name
    
    if model_dir.exists() and list(model_dir.iterdir()):
        print(f"{model_name} already exists, skipping download")
        return True
    
    model_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {model_name} ({model_info['size_gb']:.1f}GB)...")
    
    try:
        if "gdrive_folder" in model_info:
            cmd = [
                'rclone', 'copy', 
                f'gdrive:{model_info["gdrive_folder"]}',
                str(model_dir),
                '--progress'
            ]
        else:
            cmd = [
                'rclone', 'copy',
                f'gdrive:{model_info["gdrive_file"]}', 
                str(model_dir),
                '--progress'
            ]
        
        result = subprocess.run(cmd, check=True)
        print(f"{model_name} downloaded successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {model_name}: {e}")
        return False

def download_direct(model_name, model_info):
    print(f"Direct download for {model_name}...")
    print(f"Please manually download from: {model_info['gdrive_url']}")
    print(f"Extract to: models/{model_name}/")
    
    try:
        import gdown
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_dir = models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        if "gdrive_file" in model_info:
            file_id = model_info["gdrive_file"]
            output_path = model_dir / f"{model_name}.zip"
            gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output_path))
            
            if output_path.suffix == '.zip':
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                output_path.unlink()
                
            print(f"{model_name} downloaded successfully")
            return True
    except ImportError:
        print("gdown not available. Please install with: pip install gdown")
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    return False

def verify_models():
    models_dir = Path("models")
    
    for model_name in MODELS:
        model_dir = models_dir / model_name
        if not model_dir.exists():
            print(f"{model_name}: Directory not found")
            continue
            
        key_files = ["model.safetensors", "config.json"]
        missing_files = []
        
        for file_name in key_files:
            if not (model_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"{model_name}: Missing files: {missing_files}")
        else:
            print(f"{model_name}: Complete")

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for MCoT 4M")
    parser.add_argument("--method", choices=["rclone", "direct"], default="rclone",
                       help="Download method (default: rclone)")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                       help="Specific model to download (default: all)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing downloads")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_models()
        return
    
    print("MCoT 4M Model Downloader")
    print("=" * 40)
    
    models_to_download = [args.model] if args.model != "all" else list(MODELS.keys())
    
    total_size = sum(MODELS[model]["size_gb"] for model in models_to_download)
    print(f"Will download {len(models_to_download)} model(s), total size: {total_size:.1f}GB")
    
    for model in models_to_download:
        print(f"   • {model}: {MODELS[model]['description']} ({MODELS[model]['size_gb']:.1f}GB)")
    
    free_space_gb = os.statvfs('.').f_frsize * os.statvfs('.').f_bavail / (1024**3)
    if free_space_gb < total_size * 1.2:
        print(f"Warning: Only {free_space_gb:.1f}GB free space available")
        if not input("Continue anyway? (y/N): ").lower().startswith('y'):
            return
    
    if args.method == "rclone":
        if not check_rclone():
            if not install_rclone():
                print("Falling back to direct download method...")
                args.method = "direct"
            else:
                setup_rclone_gdrive()
    
    success_count = 0
    for model_name in models_to_download:
        model_info = MODELS[model_name]
        
        if args.method == "rclone":
            success = download_with_rclone(model_name, model_info)
        else:
            success = download_direct(model_name, model_info)
        
        if success:
            success_count += 1
    
    print(f"\nDownloaded {success_count}/{len(models_to_download)} models successfully")
    
    print("\nVerifying downloads...")
    verify_models()
    
    print("\nModel download complete!")
    print("\nNext steps:")
    print("1. Activate fourm environment: conda activate fourm") 
    print("2. Run VQA baseline: python fourm_inference.py --task vqa")
    print("3. Start MCoT training: python run_training_4m_mcot_fsdp.py")

if __name__ == "__main__":
    main() 