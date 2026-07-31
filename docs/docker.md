# Docker guide

This page covers the full set of Docker options for running the segmentation backend: image variants, permission handling, path mapping, and GPU setup. For the single most common command, see the [README](../README.md#docker).

## Table of contents

- [Available images](#available-images)
- [Permissions: the `--user` flag](#permissions-the---user-flag)
- [Mounting your data](#mounting-your-data)
- [Interactive shell](#interactive-shell)
- [Running as a CLI](#running-as-a-cli)
- [GPU inference](#gpu-inference)
- [Building a custom image](#building-a-custom-image)
- [Troubleshooting](#troubleshooting)

---

## Available images

| Image | Hardware | Notes |
|---|---|---|
| `dbouget/raidionics-segmenter:v1.5.0-py39-cpu` | CPU only | Default, works everywhere |
| `dbouget/raidionics-segmenter:v1.5.0-py39-cuda12.4` | GPU (CUDA 12.4) | Requires matching NVIDIA driver + `nvidia-container-toolkit` |

Pull an image with:

```bash
docker pull dbouget/raidionics-segmenter:v1.5.0-py39-cpu
```

If your machine's CUDA version doesn't match `cuda12.4`, don't try to force it — build a custom image instead (see [below](#building-a-custom-image)).

---

## Permissions: the `--user` flag

Every command in this guide includes `--user $(id -u)`. This is not optional in practice: without it, any files or folders the container creates (segmentation outputs, logs, temp files) will be owned by `root` on your host filesystem, and you won't be able to read/write/delete them without `sudo`.

`$(id -u)` resolves to your current user's numeric UID at runtime, so you don't need to hard-code it. If you know your UID ahead of time, you can substitute it directly (e.g. `--user 1000`).

---

## Mounting your data

All commands mount a local directory into the container at `/workspace/resources`:

```bash
-v /home/<username>/<resources_path>:/workspace/resources
```

Replace `/home/<username>/<resources_path>` with a real path on your machine. This directory must contain, at minimum:

- A folder with the input images you want to run inference on
- A folder with the trained model(s) to use (or let the tool auto-download them)
- A destination folder where results will be written

Everything the container needs to read or write should live somewhere under this mounted path — the container can't see anything outside it.

---

## Interactive shell

Useful for debugging, inspecting the environment, or running commands manually inside the container:

```bash
docker run --entrypoint /bin/bash \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-segmenter:v1.5.0-py39-cpu
```

---

## Running as a CLI

For direct, non-interactive inference:

```bash
docker run \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-segmenter:v1.5.0-py39-cpu \
  -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

**Path notes:** the `-c` argument must point to the config file's path *inside the container*, i.e. relative to `/workspace/resources`. Concretely, if your config lives at:

```
/home/myuser/Data/Segmentation/main_config.ini
```

and you mounted `/home/myuser/Data` as your resources path, the correct `-c` value is:

```
/workspace/resources/Segmentation/main_config.ini
```

**Verbosity levels** (`-v`): `debug`, `info`, `warning`, `error`.

---

## GPU inference

Use the CUDA-tagged image and add `--runtime=nvidia`:

```bash
docker run \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --runtime=nvidia --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-segmenter:v1.5.0-py39-cuda12.4 \
  -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

Requirements on the host:
- An NVIDIA GPU with a driver compatible with CUDA 12.4
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- The `gpu_id` parameter set correctly in your `main_config.ini`

---

## Building a custom image

If your host's CUDA version doesn't match the published `cuda12.4` image, build your own from `Dockerfile_gpu`:

1. Open `Dockerfile_gpu` in the repository root.
2. Change the base PyTorch/CUDA image tag to match your driver's supported CUDA version.
3. Build locally:

```bash
docker build -f Dockerfile_gpu -t raidionics-segmenter:custom-gpu .
```

4. Substitute `raidionics-segmenter:custom-gpu` for the image name in any command above.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Output files owned by `root` | Forgot `--user $(id -u)` |
| `CUDA error: no kernel image is available` | Driver/CUDA version mismatch — build a custom image |
| Container can't find input files | Path passed to `-c` isn't relative to `/workspace/resources`, or the resources volume wasn't mounted correctly |
| `docker: Error response from daemon: could not select device driver` | `nvidia-container-toolkit` not installed, or `--runtime=nvidia` omitted |
