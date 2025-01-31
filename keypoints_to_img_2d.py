import numpy as np
import cv2
import torch

class KeypointsToPoseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("KEYPOINTS",),
                "image_size": ("INT", {"default": 512, "min": 128, "max": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_pose"

    CATEGORY = "Pose Processing"

    def render_pose(self, keypoints, image_size):
        canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # Scale function
        def scale(pt):
            return (int(pt[0] * image_size), int(pt[1] * image_size))

        # Define keypoint connections (stickman bones)
        bones = [
            ("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
            ("neck", "left_hip"), ("neck", "right_hip"),
            ("left_hip", "left_knee"), ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
        ]

        # Draw bones
        for joint1, joint2 in bones:
            if joint1 in keypoints and joint2 in keypoints:
                cv2.line(canvas, scale(keypoints[joint1]), scale(keypoints[joint2]), (0, 0, 0), 2)

        # Draw keypoints
        for joint, pos in keypoints.items():
            cv2.circle(canvas, scale(pos), 5, (255, 0, 0), -1)

        # Convert numpy array to torch tensor and normalize
        canvas_tensor = torch.from_numpy(canvas).float() / 255.0
        # Add batch dimension (B, H, W, C)
        canvas_tensor = canvas_tensor.unsqueeze(0)
        
        return (canvas_tensor,)

class KeypointsInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "head_y": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "neck_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "neck_y": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Add all other keypoints following the same pattern
                "left_shoulder_x": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_shoulder_y": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_shoulder_x": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_shoulder_y": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Continue with remaining keypoints...
            }
        }

    RETURN_TYPES = ("KEYPOINTS",)
    FUNCTION = "get_keypoints"

    CATEGORY = "Pose Processing"

    def get_keypoints(self, **kwargs):
        keypoints = {}
        joints = [
            "head", "neck", 
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        for joint in joints:
            x = kwargs.get(f"{joint}_x", 0.5)
            y = kwargs.get(f"{joint}_y", 0.5)
            keypoints[joint] = (x, y)
            
        return (keypoints,)

NODE_CLASS_MAPPINGS = {
    "KeypointsToPoseNode": KeypointsToPoseNode,
    "KeypointsInputNode": KeypointsInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeypointsToPoseNode": "Pose Renderer (Stick Figure)",
    "KeypointsInputNode": "Pose Keypoints Input"
}
