from .hash_utils_nodes import (
    SimHashCompareNode,
    ImageSelectorNode, 
    TemporalConsistencyNode,
    ImageReferenceUpdateNode,
    FrameBlendNode
)
from .keypoint_utils_nodes import (KeypointsToPoseNode, KeypointsInputNode, PoseEstimatorNode, KeypointComparatorNode, PoseDatabaseNode, PoseDifferenceNode)
# from .llm_nodes import (LLMKeypointGeneratorNode, JSONToKeypointsNode)
NODE_CLASS_MAPPINGS = {
    "SimHashCompare": SimHashCompareNode,
    "ImageSelector": ImageSelectorNode,
    "TemporalConsistency": TemporalConsistencyNode, 
    "ImageReferenceUpdate": ImageReferenceUpdateNode,
    "FrameBlend": FrameBlendNode,
    "KeypointsToPose": KeypointsToPoseNode,
    "KeypointsInput": KeypointsInputNode,
    "PoseEstimator": PoseEstimatorNode,
    "KeypointComparator": KeypointComparatorNode,
    "PoseDatabase": PoseDatabaseNode,
    "PoseDifference": PoseDifferenceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimHashCompare": "Compare Images (SimHash)",
    "ImageSelector": "Image Selector",
    "TemporalConsistency": "Temporal Consistency",
    "ImageReferenceUpdate": "Update Image Reference",
    "FrameBlend": "Frame Blend",
    "KeypointsToPose": "Keypoints to Pose",
    "KeypointsInput": "Keypoints Input",
    "PoseEstimator": "Pose Estimation",
    "KeypointComparator": "Keypoint Comparison",
    "PoseDatabase": "Pose Database",
    "PoseDifference": "Pose Difference Visualizer"
}

# NODE_CLASS_MAPPINGS = {
#     "LLMKeypointGenerator": LLMKeypointGeneratorNode,
#     "JSONToKeypoints": JSONToKeypointsNode,
# }
