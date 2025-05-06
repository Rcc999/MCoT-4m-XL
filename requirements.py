from fourm.utils import load_safetensors
from fourm.models.fm import FM

ckpt, config = load_safetensors('/path/to/checkpoint.safetensors')
fm = FM(config=config)
fm.load_state_dict(ckpt)