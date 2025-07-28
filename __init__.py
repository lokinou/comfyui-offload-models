from .offload_recall import OffloadModel, RecallModel


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OffloadModel": OffloadModel,
    "RecallModel": RecallModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OffloadModel": "Offload Model (experimental)",
    "RecallModel": "Recall offloaded Model (experimental)",
}