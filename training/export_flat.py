from torch_flatbuffers.export import Parser
import torch
from model import UNet
from tza import Reader
import os
from pathlib import Path
from copy import deepcopy
from time import sleep
import flatbuffers
import numpy as np

base_path = Path("weights/oidn-weights")

all_f = [i for i in os.listdir(base_path.__str__()) if ".tza" in i]

torch.set_grad_enabled(False)

for i in all_f:
    red = Reader(Path(base_path, i).__str__())
    in_channels = red._table["enc_conv0.weight"][0][1]
    out_channels = red._table["dec_conv0.weight"][0][2]
    model = UNet(in_channels=in_channels, out_channels=out_channels)

    data = torch.rand(1, in_channels, 32, 32)
    np.save(f"test_tensor/in_{i.replace('.tza', '.npy')}", data.permute(0,2,3,1).cpu().numpy())

    model.load_state_dict({k: torch.from_numpy(red[k][0].copy()) for k in red._table.keys()})

    out = model(data)
    np.save(f"test_tensor/out_{i.replace('.tza', '.npy')}", out.permute(0,2,3,1).cpu().numpy())
    parser = Parser(save_path="flatbuffer", name=i.replace(".tza", ""))
    parser.parse_module(model, name="denoiser")
    parser.save_to_flatbuff()
