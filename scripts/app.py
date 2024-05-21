import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Dict

import uvicorn
from dino import DINO
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from params import shared_params
from PIL import Image
from schema import DinoRequestBody
from utils import AsyncExecutor

NUM_MODELS = 2
ROOT = Path("/opt/program")
CONFIG_PATH = ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_PATH = ROOT / "weights/groundingdino_swint_ogc.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    shared_params.exec = AsyncExecutor(
        loop=asyncio.get_event_loop(),
        executor=ThreadPoolExecutor(max_workers=NUM_MODELS),
    )
    init_model_coroutines = [
        shared_params.exec(DINO, CONFIG_PATH, MODEL_PATH) for _ in range(NUM_MODELS)
    ]
    models = await asyncio.gather(*init_model_coroutines)
    shared_params.models.extend(models)
    print("ALL Model is ready.")
    yield
    print("End")


app = FastAPI(lifespan=lifespan)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to perform blocking tasks
def run_blocking_tasks(item: DinoRequestBody):
    model = shared_params.models.pop(0)
    shared_params.models.append(model)
    image_data = base64.b64decode(item.image)
    image_source = Image.open(BytesIO(image_data)).convert("RGB")
    return model.inference(image_source, item.prompt)


@app.post("/inference", response_model=Dict[str, object])
async def annotate_image(dino_request: DinoRequestBody):
    response_data = await shared_params.exec(run_blocking_tasks, dino_request)
    json_compatible_data = jsonable_encoder(response_data)
    return JSONResponse(content=json_compatible_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the FastAPI application with Uvicorn."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host to bind to."
    )
    parser.add_argument("--port", type=int, default=9001, help="The port to bind to.")
    parser.add_argument(
        "--workers", type=int, default=2, help="The number of worker processes."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "app:app", host=args.host, port=args.port, reload=False, workers=args.workers
    )
