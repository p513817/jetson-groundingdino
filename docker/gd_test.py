import torch
from groundingdino.util.inference import load_model

model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth",
)
model = model.to("cuda:0")
print(torch.cuda.is_available())
print("DONE!")
