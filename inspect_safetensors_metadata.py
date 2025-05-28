from safetensors.torch import load_file
import json

def inspect_metadata(file_path="vqa.safetensors"):
    print(f"Attempting to inspect metadata for: {file_path}")
    try:
        # Load the full metadata. If the file is very large and you only need metadata,
        # there isn't a direct way to avoid loading the tensors first with load_file,
        # but it returns metadata separately.
        # The load_safetensors utility in fourm.utils.checkpoint already does this.
        
        # We are interested in the raw metadata dictionary returned by safetensors.torch.load_file
        # when return_metadata=True is (implicitly or explicitly) handled by a higher-level loader.
        # For a direct look, we can use load_file and check its second return value if a utility isn't handy.
        # However, the user's fourm.utils.checkpoint.load_safetensors *already* tries to get this.
        # Its return is state_dict, config_from_ckpt
        # config_from_ckpt = loaded_metadata.get("__metadata__", {}).get("args")
        # This implies that the raw metadata IS being loaded, but either __metadata__ is missing,
        # or args under it is missing.

        # Let's use the most direct way to get the top-level metadata from the file.
        # The safetensors format stores metadata as a JSON header.
        # We can open the file and read this header part.
        # This avoids loading all tensors if we just want to see the raw metadata string.

        with open(file_path, 'rb') as f:
            # First 8 bytes are the size of the metadata JSON.
            metadata_len_bytes = f.read(8)
            if not metadata_len_bytes:
                print("Error: Could not read metadata length. File might be empty or too small.")
                return
            
            metadata_len = int.from_bytes(metadata_len_bytes, 'little')
            
            # Read the metadata JSON string itself.
            metadata_json_str = f.read(metadata_len)
            if not metadata_json_str:
                print("Error: Could not read metadata JSON. Length might be incorrect or file corrupted.")
                return

        # Decode the JSON string.
        raw_metadata = json.loads(metadata_json_str.decode('utf-8'))

        print("\n--- Raw Safetensors Metadata ---")
        if raw_metadata:
            for k, v in raw_metadata.items():
                if k == "__metadata__":
                    print(f"  Found '__metadata__' key:")
                    if isinstance(v, dict):
                        for nk, nv in v.items():
                            print(f"    {nk}: {nv}") # Print with limited depth for readability
                            if isinstance(nv, dict): 
                                print(f"      (Value is a dict, not showing full content if too large)")
                            elif isinstance(nv, list):
                                print(f"      (Value is a list, showing first few items if large: {str(nv)[:200]}...)")
                    else:
                        print(f"    __metadata__: {v}") # Should be a dict
                else:
                    # For other top-level keys, print them directly.
                    # These usually describe tensor layouts, not config.
                    print(f"  {k}: (tensor description, not shown)") 
            
            if "__metadata__" not in raw_metadata:
                print("  '__metadata__' key NOT FOUND in the raw metadata.")

        else:
            print("  Raw metadata block is empty or None.")
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Assuming vqa.safetensors is in the same directory as this script,
    # or in the workspace root if you run from there.
    inspect_metadata(file_path="vqa.safetensors") 