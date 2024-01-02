#!/bin/bash
docker run  --rm --gpus all \
-v "/media/uofsko-lab/c61d4ea0-b8fe-4205-9815-f7d3d054043c/hamis/DP-LLM":"/workspace" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY --net=host -it hamis_dpllm
# --user uofsko-lab:1000 \
