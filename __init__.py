from .image_simhash_nodes import (
    SimHashCompareNode,
    ImageSelectorNode, 
    TemporalConsistencyNode,
    ImageReferenceUpdateNode,
    FrameBlendNode
)

NODE_CLASS_MAPPINGS = {
    "SimHashCompare": SimHashCompareNode,
    "ImageSelector": ImageSelectorNode,
    "TemporalConsistency": TemporalConsistencyNode, 
    "ImageReferenceUpdate": ImageReferenceUpdateNode,
    "FrameBlend": FrameBlendNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimHashCompare": "Compare Images (SimHash)",
    "ImageSelector": "Image Selector",
    "TemporalConsistency": "Temporal Consistency",
    "ImageReferenceUpdate": "Update Image Reference",
    "FrameBlend": "Frame Blend"
}
