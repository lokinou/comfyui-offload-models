# ComfyUI Offload models

Manually offload and reload models between devices (cpu, cuda ...).
Useful when restricted on VRAM. 

## Usage

Place the nodes around the model you want to execute, connecting an input through (i.e. generated image) that serves as trigger.

- "Offload Model" after processing, allowing to send the model to cpu
- "Recall Model" sending back the model to cuda

Requires a bit of control on the workflow execution, it can be more convenient than deleting models and their cache.

## Getting started

### Install

#### Via ComfyUI

Search for "comfyui-offload-models" in the available node, or paste the github repo link to install from it. 

#### Manual Install
```sh
# Go to the custom nodes
cd ./ComfyUI/custom_nodes
# Install the repo
git clone https://github.com/lokinou/comfyui-offload-models.git
```

## Updates

* `0.0.1` : First version uploaded, with an Experimental flag

## Examples

TODO