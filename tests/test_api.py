#!/usr/bin/python3

import argparse
import base64
import concurrent
import concurrent.futures
import os
from pathlib import Path

import cv2
import requests


def get_current_folder() -> Path:
    return Path(__file__).parents[0]


def cv2base64(image):
    retval, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer)


def draw_results(image, results):
    for result in results:
        bboxes, conf, label = result
        cx, cy, width, height = bboxes
        cx *= image.shape[1]
        cy *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]
        x = int(cx - width / 2)
        y = int(cy - height / 2)
        w = int(width)
        h = int(height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(
            image,
            f"{label}: {conf}",
            (x, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def block_event(host, port, route, index: int = 0):
    prompt = "where is the cat?"
    image = cv2.imread(os.path.join(get_current_folder(), "./cats.jpg"))
    data = {"prompt": prompt, "image": cv2base64(image)}
    llm_url = f"http://{host}:{port}{route}"
    response = requests.post(llm_url, json=data)
    results = response.json()
    image = draw_results(image=image, results=results)
    cv2.imwrite(f"output-{index}.jpg", image)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the FastAPI application with Uvicorn."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="The host to bind to."
    )
    parser.add_argument("--port", type=int, default=9009, help="The port to bind to.")
    parser.add_argument(
        "--route", type=str, default="/inference", help="The api route."
    )

    return parser.parse_args()


if __name__ == "__main__":
    import time

    args = parse_args()
    futures = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # For pressure testing
        for i in range(1):
            futures[
                executor.submit(
                    block_event,
                    host=args.host,
                    port=args.port,
                    route=args.route,
                    index=i,
                )
            ] = time.time()

        for future in concurrent.futures.as_completed(futures):
            created_time = futures[future]
            print(future.result())
            print(f"cost: {time.time()-created_time}")
