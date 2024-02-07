from briarmbg import BriaRMBG
import torch
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')

net = BriaRMBG()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()

# push to hub
net.push_to_hub("nielsr/RMBG-1.4")

# reload
net = BriaRMBG.from_pretrained("nielsr/RMBG-1.4")