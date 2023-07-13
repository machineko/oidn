from torch_flatbuffers.export import Parser
import torch
from model import UNet
from tza import Reader
import os
from pathlib import Path
from copy import deepcopy

base_path = Path("weights/oidn-weights")

all_f = [i for i in os.listdir(base_path.__str__()) if ".tza" in i]



for i in all_f:
    red = Reader(Path(base_path, i).__str__())
    in_channels = red._table["enc_conv0.weight"][0][1]
    out_channels = red._table["dec_conv0.weight"][0][2]
    model = UNet(in_channels=in_channels, out_channels=out_channels)

    model.load_state_dict({k: torch.from_numpy(red[k][0].copy()) for k in red._table.keys()})
    parser = Parser(save_path="flatbuffer", name=i.replace(".tza", ""))
    parser.parse_module(module=deepcopy(model), name="denoiser")
    parser.save_to_flatbuff()