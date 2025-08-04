import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher
from typing import Tuple, List, Union
from dataclasses import dataclass
import torch
import gc
import logging

logger = logging.getLogger(__name__)

# Note: This doesn't work with reroute for some reason?
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


@dataclass
class ModelInfo:
    classname: str
    device_current: Union[torch.device, int]
    device_target: Union[torch.device, int]
    device_offload: Union[torch.device, int]
    move_func: callable  # function to call to change the device


any = AnyType("*")

UNSUPPORTED_CHK = [
    (
        ['model', 'diffusion_model', 'model'],
        "NunchakuFluxTransformer2dModel",
        "Nunchaku not supported (offloading managed in the binaries).\n"
        "Disable this offload node, and use the option to enable/disable automatic offloading in the nunchaku loader.",
    ),
    (
        ['diffusion_model', 'model'],
        "NunchakuFluxTransformer2dModel",
        "Nunchaku not supported (offloading managed in the binaries).\n"
        "Disable this offload node, and use the option to enable/disable automatic offloading in the nunchaku loader.",
    ),
]


device_options = ["auto","cpu"]
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        # This creates user-friendly names like "cuda:0"
        device_name = torch.cuda.get_device_name(i)
        #device_options.append(f"cuda:{i} ({device_name})")  # People should know already their devices :)
        device_options.append(f"{torch.device(i)}")

class OffloadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any, )}, # For passthrough
            "optional": {"model": (any, ),
                         "device": (device_options, {"default": "auto", "label": "Load Device", "tooltip": "Select the device to offload the model to."}),
                         "on_error": (["ignore", "raise"], {"default": "raise", "label": "On Error", "tooltip": "What to do on error: ignore or raise an exception."}),
                         "enable": ("BOOLEAN", {"default": True, "label": "Enable Offload", "tooltip": "Enable offloading of the model to the offload device."})                         
                         },
        }
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    RETURN_TYPES = (any, any)
    FUNCTION = "route"
    CATEGORY = "Unload Model"
    
    def route(self, **kwargs):
        logging.info("Offload Model (node)")
        model_candidate = kwargs.get("model")
        
        if not kwargs.get("enable", True):
            return (kwargs.get("value"), kwargs.get("model"),)

        # Check if the model is valid
        if not is_supported(model_candidate=model_candidate, on_error=kwargs.get("on_error", "raise"))[0]:
            return (kwargs.get("value"), kwargs.get("model"),)

        # get the device and function do move it between devices
        list_models = scan_for_models(top_model=model_candidate)
        for model in list_models:
            m_info: ModelInfo = get_model_info(model)
            cls = m_info.classname
            #preferred_device = m_info.device_target if m_info.device_target is not None else mm.get_torch_device()
            
            if kwargs.get("device", "auto") == "auto":
                offload_device = mm.unet_offload_device() if m_info.device_offload is not None else mm.unet_offload_device()
            else:
                # Use the requested device from parameters
                offload_device = torch.device(kwargs.get("device"))

            if torch.device(m_info.device_current) != torch.device(offload_device):

                logging.info(f'- Offload {cls}: move from {torch.device(m_info.device_current)}'
                      f' to {torch.device(offload_device)}...')
                m_info.move_func(torch.device(offload_device))
                logging.info(f'- Offload {cls}: done')

            # Validate the migration
            m_info_post: ModelInfo = get_model_info(model)
            if torch.device(m_info_post.device_current) == torch.device(offload_device):
                logging.info(f'- Offload {cls}: validated')
                logging.debug('- Freeing VRAM...')
                gc.collect()
                mm.cleanup_models_gc()
                mm.soft_empty_cache()
                logging.debug('- cleanup done')
                # todo custom cleanup for known models? eg. flux transformer
                # model_size = mm.module_size(self.transformer)
                # do migration to offload device
                # mm.free_memory(model_size, device)
            else:
                logging.error(f'- Error for {cls}: Could not validate offloading, '
                      f'model is on {torch.device(m_info_post.device_current)} instead of {torch.device(offload_device)}')

        return (kwargs.get("value"), kwargs.get("model"),)
    
    
class RecallModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any, )}, # For passthrough
            "optional": {"model": (any, ),
                         "device": (device_options, {"default": "auto", "label": "Load Device", "tooltip": "Select the device to recall the model to."}),
                         "on_error": (["ignore", "raise"], {"default": "raise", "label": "On Error", "tooltip": "What to do on error: ignore or raise an exception."}),
                         "enable": ("BOOLEAN", {"default": True, "label": "Enable Recall", "tooltip": "Enable recall of the model to the preferred device."}),
                         
                         },
        }
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    RETURN_TYPES = (any, any)
    FUNCTION = "route"
    CATEGORY = "Unload Model"

    def route(self, **kwargs):
        logging.info("Recall Model (node)")
        model_candidate = kwargs.get("model")
        cls = model_candidate.__class__.__name__
        if not kwargs.get("enable", True):
            return (kwargs.get("value"), kwargs.get("model"),)

        # Check if the model is valid
        
        if not is_supported(model_candidate=model_candidate, on_error=kwargs.get("on_error", "raise"))[0]:
            return (kwargs.get("value"), kwargs.get("model"),)
            

        # get the device and function do move it between devices
        list_models = scan_for_models(top_model=model_candidate)
        if len(list_models) > 0:
            logging.debug('- Freeing VRAM...')
            mm.soft_empty_cache()
            gc.collect()
            logging.debug('- done')
        for model in list_models:
            m_info: ModelInfo = get_model_info(model)
            
            if kwargs.get("device", "auto") == "auto":
                preferred_device = m_info.device_target if m_info.device_target is not None else mm.get_torch_device()
            else:
                # Use the requested device from parameters
                preferred_device = torch.device(kwargs.get("device"))
            #offload_device = mm.unet_offload_device() if m_info.device_offload is not None else mm.unet_offload_device()

            if torch.device(m_info.device_current) != torch.device(preferred_device):
                logging.info(f'- Recall {cls} from {torch.device(m_info.device_current)}'
                      f' to {torch.device(preferred_device)}...')
                m_info.move_func(torch.device(preferred_device))
                logging.info(f'- Recalling {cls} done')

            # Validate the migration
            m_info_post: ModelInfo = get_model_info(model)
            if torch.device(m_info_post.device_current) == torch.device(preferred_device):
                logging.info(f'- Recalling {cls} validated')
            else:
                logging.error(f'- Error for {cls}: Could not validate recall, '
                      f'model is on {torch.device(m_info_post.device_current)} instead of {torch.device(preferred_device)}')

        return (kwargs.get("value"), kwargs.get("model"),)

def is_supported(model_candidate, on_error: str = "raise") -> Tuple[bool, str]:
    """
    Return true if the model is supported
    """
    # Eclude unsupported models first
    for nested_obj, class_name, err_msg in UNSUPPORTED_CHK:
        # Check for unsupported models
        if get_nested_class_name(obj=model_candidate, path=nested_obj) == class_name:
            err_str = f"Unsupported {model_candidate.__class__.__name__} model.\n {err_msg}"
            logging.error(f"- Error: {err_str}")
            if on_error == "raise":
                raise ValueError(err_str)
            else:
                return False, err_str
        

    # Then by default check for supported models
    if type(model_candidate) == ModelPatcher:
        return True, ''
    elif issubclass(type(model_candidate), ModelPatcher):
        logging.info(f"- model of type {model_candidate.__class__.__name__} might not be supported for Offload/recall")
        return True, ''
    elif hasattr(model_candidate, 'device') and hasattr(model_candidate, 'to'):
        logging.info(f"- Model of type {model_candidate.__class__.__name__} supported (contains 'model.device' and 'model.to()')")
        return True, ''
    else:

        # If no checks matched, log a warning   
        logging.warning(f"- Warning: No compatible device found for this model {model_candidate.__class__.__name__}.")
        return False


def scan_for_models(top_model: object) -> List[object]:
    """
    Return supported models, and eventually embedded models
    Args:
        model: The model to check.
    Returns:
        List[object]: the current model if supported and any embedded one (e.g. ModelPatcher contains a model)
    """
    if type(top_model) == ModelPatcher or issubclass(type(top_model), ModelPatcher):
        return [top_model, top_model.model]
    elif hasattr(top_model, 'device') and hasattr(top_model, 'to'):
        return [top_model]
    else:
        return []
    

def get_nested_class_name(obj, path):
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return getattr(obj.__class__, '__name__', None)


def get_model_info(model) -> ModelInfo:
    """
    Get info about the model and its devices
    Args:
        model: The model to check.
    Returns:
        ModelInfo: info summary about the devices

    """
    if type(model) == ModelPatcher or issubclass(type(model), ModelPatcher):
        # model patcher
        mp_info = ModelInfo(classname=type(model).__name__,
                       device_current=model.current_loaded_device(),
                       device_target=model.load_device,
                       device_offload=model.offload_device if hasattr(model, 'offload_device') else None,
                       move_func=model.model_patches_to)
        return mp_info

    else:
        m_info = ModelInfo(classname=type(model).__name__,
                       device_current=model.device,
                       device_target=None,
                       device_offload=model.offload_device if hasattr(model, 'offload_device') else None,
                       move_func=model.to)
        return m_info