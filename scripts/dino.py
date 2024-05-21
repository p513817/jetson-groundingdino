from pathlib import Path

import groundingdino.datasets.transforms as T
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image


class DINO:
    def __init__(
        self,
        config_path: str,
        model_path: str,
        warm_up: int = 2,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        assert self.model_path.exists(), "model is not exists"
        assert self.config_path.exists(), "config is not exists"

        self.model = load_model(config_path, model_path)
        self.box_threshold = 0.35
        self.text_threshold = 0.25

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self._warm_up_event(iterations=warm_up)

    def preprocess(self, frame: Image):
        return self.transform(frame, None)[0]

    def postprocess(self, results: list):
        boxes, confidences, labels = (
            results[0].tolist(),
            results[1].tolist(),
            results[2],
        )
        assert len(boxes) == len(confidences) == len(labels), "shape not the same"
        return [list(item) for item in zip(boxes, confidences, labels)]

    def inference(self, frame: torch.Tensor, prompt: str):
        return self.postprocess(
            results=predict(
                model=self.model,
                image=self.preprocess(frame),
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        )

    def _warm_up_event(self, iterations: int = 4):
        image_source = Image.new("RGB", (100, 100), color="white")
        for _ in range(iterations):
            self.inference(image_source, "where is the white background")
