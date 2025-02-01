from .image_simhash_nodes import (
    SimHashCompareNode,
    ImageSelectorNode, 
    TemporalConsistencyNode,
    ImageReferenceUpdateNode,
    FrameBlendNode
)
from .keypoints_to_img_2d import (KeypointsToPoseNode, KeypointsInputNode)

NODE_CLASS_MAPPINGS = {
    "SimHashCompare": SimHashCompareNode,
    "ImageSelector": ImageSelectorNode,
    "TemporalConsistency": TemporalConsistencyNode, 
    "ImageReferenceUpdate": ImageReferenceUpdateNode,
    "FrameBlend": FrameBlendNode,
    "KeypointsToPose": KeypointsToPoseNode,
    "KeypointsInput": KeypointsInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimHashCompare": "Compare Images (SimHash)",
    "ImageSelector": "Image Selector",
    "TemporalConsistency": "Temporal Consistency",
    "ImageReferenceUpdate": "Update Image Reference",
    "FrameBlend": "Frame Blend",
    "KeypointsToPose": "Keypoints to Pose",
    "KeypointsInput": "Keypoints Input"
}
