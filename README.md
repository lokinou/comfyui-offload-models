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
* `0.0.3` : Fixed a faulty condition, improved node names, individualized modelpatcher and model which may be on different devices at once
* `0.0.2` : Both offload the modelpatcher and the referenced model, otherwise it doesn't free VRAM
* `0.0.1` : First version uploaded, with an Experimental flag

## Examples

TODO