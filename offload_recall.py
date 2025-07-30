import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher
import gc
from typing import Tuple, List


# Note: This doesn't work with reroute for some reason?
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class OffloadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any, )}, # For passthrough
            "optional": {"model": (any, )},
        }
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    RETURN_TYPES = (any, any)
    FUNCTION = "route"
    CATEGORY = "Unload Model"
    
    def route(self, **kwargs):
        print("Offload Model:")

        model_candidate = kwargs.get("model")
        # Check if the model is valid
        if not check_model(model=model_candidate, action_desc="Offload"):
            return (kwargs.get("value"), kwargs.get("model"),)

        print(f" - Model of type {type(model_candidate).__class__.__name__} found.")
        # get the device and function do move it between devices
        list_devices_to_move = get_device(model_candidate)
        preferred_device = mm.get_torch_device()  # default device (i.e. most likely cuda)
        offload_device = mm.unet_offload_device()  # offload device (i.e. most likely cpu)

        off_device = [dev == offload_device for dev, _ in list_devices_to_move]
        if all(off_device):
            print(f" - model already on the offload device {preferred_device}, nothing to do")
            return (kwargs.get("value"), kwargs.get("model"),)
        else:
            print(f" - offloading the model of type {type(model_candidate)}")
            print(f" - [1] gc.collect()")
            gc.collect()  # run garbage collection
            for i, (current_device, move_func) in enumerate(list_devices_to_move):
                if not off_device[i]:
                    print(f" - [2-{i}] offloading from {current_device} to {offload_device} using {move_func.__name__}")
                    move_func(offload_device)
            print(f" - [3] free VRAM cache")
            mm.soft_empty_cache()  # clear cache after offloading
            print(f" - [4] gc.collect()")
            gc.collect()  # run garbage collection
            print(f" - offloading done")
            
        return (kwargs.get("value"), kwargs.get("model"),)
    
    
class RecallModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any, )}, # For passthrough
            "optional": {"model": (any, )},
        }
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    RETURN_TYPES = (any, any)
    FUNCTION = "route"
    CATEGORY = "Unload Model"
    
    def route(self, **kwargs):
        print("Rapatriate Offloaded Model:")
        model_candidate = kwargs.get("model")
        # Check if the model is valid
        if not check_model(model=model_candidate, action_desc="Rapatriate/Load"):
            return (kwargs.get("value"), kwargs.get("model"),)

        print(f" - Model of type {model_candidate.__class__.__name__} found.")
        # get the device and function do move it between devices
        list_devices_to_move = get_device(model_candidate)

        preferred_device = mm.get_torch_device()  # default device (i.e. most likely cuda)
        offload_device = mm.unet_offload_device()  # offload device (i.e. most likely cpu)

        on_device = [dev == preferred_device for dev, _ in list_devices_to_move]
        if all(on_device):
            print(f" - Model already on the Preferred device {preferred_device}, nothing to do")
            return (kwargs.get("value"), kwargs.get("model"))
        else:
            print(f" - rapatriating the model of type {model_candidate.__class__.__name__}")
            print(f" - [1] free VRAM cache")
            mm.soft_empty_cache()  # clear cache after offloading
            print(f" - [2] gc.collect()")
            gc.collect()  # run garbage collection
            for i, (current_device, move_func) in enumerate(list_devices_to_move):
                if not on_device[i]:
                    print(f" - [3-{i}] rapatriating from {current_device} to {preferred_device} using {move_func.__name__}")
                    move_func(preferred_device)
            print(f" - [4] gc.collect()")
            print(f" - done rapatriating")
            
        return (kwargs.get("value"), kwargs.get("model"),)        # {"ui": {"text": (value,)}}

def check_model(model, action_desc: str = "Offload/Onload")-> bool:
    """
    Check if the model has a device and a method to move it to a device.
    Args:
        model: The model to check.
    Returns:
        pass: (bool): True if the model is supported, False otherwise.
    """
    if model is None:
        print(f"- Warning: No model provided (None object)")
        return False
    elif hasattr(model, 'model') and not hasattr(model, 'device') and not hasattr(model, 'to'):
        if not hasattr(model, 'model_patches_to'):
            print(f"- Warning: Cannot {action_desc} model. Model of type {model.__class__.__name__} contains a submodel {type(model.model).__class__} that has no specified 'model_patches_to'  method")
            return False
        if not hasattr(model.model, 'device'):
            print(f"- Warning: Cannot {action_desc} model. Model of type {model.__class__.__name__} contains a submodel {type(model.model).__class__} that has no specified 'device'")
            return False
    elif not hasattr(model, 'device'):
        # This is a model, unload it
        print(f"- Warning: Cannot {action_desc} model. Model of type {model.__class__.__name__} has no specified 'device'")
        return False
    elif not hasattr(model, 'to'):
        print(f"- Warning: Cannot {action_desc} model. Model of type {model.__class__.__name__} has no 'to' method, ")
        return False

        
    return True

def get_device(model) -> Tuple[str, List[callable]]:
    """
    Get the device of the model and the function to move it to a device.
    Args:
        model: The model to check.
    Returns:
        List[Tuple[str, callable]]: A list of tuples containing the current device and the function to move the model to a device.
    """
    ret = []
    if type(model) == ModelPatcher:
        ret.append((model.load_device, model.model_patches_to))
        ret.append((model.model.device, model.model.to))
    elif issubclass(type(model), ModelPatcher):
        print(f"- Model of type {model.__class__.__name__} is an implementation of ModelPatcher, assuming it has a 'model_patches_to' method")
        ret.append((model.load_device, model.model_patches_to))
        ret.append((model.model.device, model.model.to))
    else:
        ret.append((model.device, model.to))

    return ret
