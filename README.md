# Jetson GroundingDINO Service
GroundingDINO for Jetson with Model Service API

[![alt text](https://img.youtube.com/vi/_r80q5bV87Q/0.jpg)](https://www.youtube.com/watch?v=_r80q5bV87Q)

## Features
* Docker Image for Jetson Platform
* Model Service with GroundingDINO

## Setup ( Important! )
* sudo vim /etc/docker/daemon.json
* new content
    ```json
        {
            "runtimes": {
                "nvidia": {
                    "path": "nvidia-container-runtime",
                    "runtimeArgs": []
                }
            },

            "default-runtime": "nvidia"
        }
    ```
* restart service
    ```bash
    sudo systemctl restart docker
    ```

## Params
```bash
export IMAGE_NAME="jetson-groundingdino:v0.0.1-beta"
export CNTR_NAME="jgd"
export JGD_PORT=9009
```

## Build Docker Image
```bash
sudo docker build \
-t ${IMAGE_NAME} \
-f ./docker/Dockerfile .
```

## Run Docker Image
```bash
sudo docker run \
--runtime nvidia \
-it --rm \
--name ${CNTR_NAME} \
-p ${JGD_PORT}:${JGD_PORT} \
-v $(pwd)/scripts:/workspace \
${IMAGE_NAME} \
python3 app.py --port ${JGD_PORT}
```

## Reference
* [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
