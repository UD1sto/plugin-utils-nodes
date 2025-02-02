import numpy as np
import cv2
import torch
from annoy import AnnoyIndex
import mediapipe as mp
import os
import pickle

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
            ("head", "neck"), ("left_eye", "right_eye"), 
            ("left_eye", "left_ear"), ("right_eye", "right_ear"),
            ("left_ear", "neck"), ("right_ear", "neck"),
            
            # Upper body
            ("neck", "left_shoulder"), ("neck", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_hand"), ("right_elbow", "right_hand"),
            
            # Lower body
            ("neck", "left_hip"), ("neck", "right_hip"),
            ("left_hip", "right_hip"),  # Pelvis connection
            ("left_hip", "left_knee"), ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
            ("left_ankle", "left_heel"), ("right_ankle", "right_heel"),
            ("left_heel", "left_toe"), ("right_heel", "right_toe"),
            
            # Side body connections
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("neck", "mid_hip"),  # Main spinal connection
            ("mid_hip", "left_hip"), ("mid_hip", "right_hip"),
            ("left_elbow", "left_hand"), 
            ("right_elbow", "right_hand"),
            ("neck", "mid_back"),
            ("mid_back", "mid_hip"),
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
                "left_shoulder_x": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_shoulder_y": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_shoulder_x": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_shoulder_y": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_eye_x": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_eye_y": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_eye_x": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_eye_y": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_ear_x": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_ear_y": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_ear_x": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_ear_y": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_toe_x": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_toe_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_toe_x": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_toe_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_hand_x": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "left_hand_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_hand_x": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right_hand_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mid_hip_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mid_hip_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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
            "left_hand", "right_hand",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_eye", "right_eye",
            "left_ear", "right_ear",
            "left_toe", "right_toe",
            "left_heel", "right_heel",
            "mid_hip",
        ]
        
        for joint in joints:
            x = kwargs.get(f"{joint}_x", 0.5)
            y = kwargs.get(f"{joint}_y", 0.5)
            keypoints[joint] = (x, y)
            
        return (keypoints,)

class PoseEstimatorNode:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "det_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("KEYPOINTS",)
    FUNCTION = "estimate_pose"
    CATEGORY = "Pose Processing"

    def estimate_pose(self, image, det_threshold):
        # Convert tensor to numpy
        image_np = image.cpu().numpy().squeeze() * 255
        image_np = image_np.astype(np.uint8)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(image_rgb)
        
        keypoints = {}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility > det_threshold:
                    # Map MediaPipe landmarks to our naming convention
                    keypoint_name = self._get_keypoint_name(idx)
                    if keypoint_name:
                        keypoints[keypoint_name] = (landmark.x, landmark.y)
        
        print(f"Detected {len(keypoints)} keypoints with confidence > {det_threshold}")
        
        keypoints = {k: (max(0, min(1, x)), max(0, min(1, y))) 
                     for k, (x,y) in keypoints.items()}
        
        return (keypoints,)
    
    def _get_keypoint_name(self, idx):
        # MediaPipe pose landmark mapping
        return {
            0: "head",  # MediaPipe's nose -> our head
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_hand",
            16: "right_hand",
            23: "left_heel",
            24: "right_heel",
            25: "left_toe",
            26: "right_toe",
            27: "mid_hip",  # MediaPipe's pelvis
            28: "mid_back",  # Additional spine point
            17: "left_hip",
            18: "right_hip",
            19: "left_knee",
            20: "right_knee",
            21: "left_ankle",
            22: "right_ankle",
        }.get(idx, None)  # Return None for unmapped points

class KeypointComparatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_keypoints": ("KEYPOINTS",),
                "target_keypoints": ("KEYPOINTS",),
                "comparison_mode": (["positional", "vector_angle"], {"default": "positional"}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "compare_keypoints"
    CATEGORY = "Pose Processing"

    def compare_keypoints(self, source_keypoints, target_keypoints, comparison_mode):
        # Weighting for different body parts
        JOINT_WEIGHTS = {
            "head": 1.2,
            "neck": 1.5,
            "shoulders": 1.1,
            "hands": 0.9,
            "hips": 1.0
        }
        
        total = 0
        matched_joints = 0
        
        for joint in source_keypoints:
            if joint not in target_keypoints:
                continue
            
            # Get weight based on joint type
            weight = 1.0
            for k,v in JOINT_WEIGHTS.items():
                if k in joint:
                    weight = v
                    break
                
            # Positional difference
            dx = source_keypoints[joint][0] - target_keypoints[joint][0]
            dy = source_keypoints[joint][1] - target_keypoints[joint][1]
            dist = (dx**2 + dy**2) * weight
            
            # Angular difference for limbs
            if "angle" in comparison_mode:
                # Calculate bone vectors
                if "elbow" in joint:
                    upper = ... # Calculate vector from shoulder to elbow
                    lower = ... # Calculate vector from elbow to wrist
                    angle_diff = angle_between(upper, lower)
                    dist += angle_diff * 0.1
                
            total += dist
            matched_joints += 1
        
        return total / max(1, matched_joints)

class PoseDatabaseNode:
    def __init__(self):
        self.db_dir = os.path.join(os.path.dirname(__file__), "pose_database")
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize vector length FIRST
        self.vector_length = len(self.ORDERED_JOINTS) * 2
        self.index = None
        self.metadata = []
        
        self._init_index()  # Now has access to vector_length

    @property
    def ORDERED_JOINTS(self):
        return [
            "head", "neck", 
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_hand", "right_hand",
            "left_hip", "right_hip", "mid_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_heel", "right_heel",
            "left_toe", "right_toe",
            "left_eye", "right_eye",
            "left_ear", "right_ear"
        ]

    def _init_index(self):
        index_path = os.path.join(self.db_dir, "index.ann")
        metadata_path = os.path.join(self.db_dir, "metadata.pkl")
        
        # Create new writable index instance
        self.index = AnnoyIndex(self.vector_length, 'angular')
        self.metadata = []
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing data into temporary index
            temp_index = AnnoyIndex(self.vector_length, 'angular')
            temp_index.load(index_path)
            
            # Copy items to new index
            for i in range(temp_index.get_n_items()):
                self.index.add_item(i, temp_index.get_item_vector(i))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Verify alignment
            if len(self.metadata) != self.index.get_n_items():
                raise ValueError("Index/metadata count mismatch. Delete database folder and restart")

    def _save_database(self):
        index_path = os.path.join(self.db_dir, "index.ann")
        metadata_path = os.path.join(self.db_dir, "metadata.pkl")
        
        # Create new index for saving
        save_index = AnnoyIndex(self.vector_length, 'angular')
        for i in range(self.index.get_n_items()):
            save_index.add_item(i, self.index.get_item_vector(i))
        
        save_index.build(10)
        save_index.save(index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _keypoints_to_vector(self, kp):
        vector = []
        for joint in self.ORDERED_JOINTS:  # Use the property
            x, y = kp.get(joint, (0,0))
            vector.extend([x, y])
        return vector

    def compare_keypoints(self, kp1, kp2):
        # Simple Euclidean distance comparison
        total = 0
        for joint in set(kp1.keys()).union(kp2.keys()):
            x1, y1 = kp1.get(joint, (0,0))
            x2, y2 = kp2.get(joint, (0,0))
            total += ((x1-x2)**2 + (y1-y2)**2)
        return total ** 0.5  # Return actual distance

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["store", "query"], {"default": "store"}),
                "keypoints": ("KEYPOINTS",),
            },
            "optional": {
                "image": ("IMAGE",)  # Only required for store action
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("best_match", "similarity_score")
    FUNCTION = "handle_pose"
    CATEGORY = "Pose Processing"

    def handle_pose(self, action, keypoints, image=None):
        self._init_index()  # Reinitialize index every time
        
        if action == "store":
            if image is None:
                raise ValueError("Image is required for store action")
            
            # Add new item to fresh index
            vec = self._keypoints_to_vector(keypoints)
            self.index.add_item(len(self.metadata), vec)
            self.metadata.append(image)
            
            # Build and save
            self.index.build(10)
            self._save_database()
            
            return (image, 1.0)
            
        else:  # Query
            if not self.metadata:
                return (torch.zeros((1, 512, 512, 3)), 0.0)
            
            vec = self._keypoints_to_vector(keypoints)
            self.index.build(10)  # Ensure index is built
            
            indices, distances = self.index.get_nns_by_vector(vec, 1, include_distances=True)
            
            if not indices:
                return (torch.zeros((1, 512, 512, 3)), 0.0)
            
            best_idx = indices[0]
            best_image = self.metadata[best_idx]
            similarity = 1 - (distances[0] / 2)
            
            return (best_image, similarity)

    def _vector_to_keypoints(self, idx):
        # Implementation of _vector_to_keypoints method
        pass

class PoseDifferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_a": ("IMAGE",),
                "pose_b": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_difference"
    CATEGORY = "Pose Processing"

    def visualize_difference(self, pose_a, pose_b, blend_factor):
        # Convert tensors to numpy arrays
        img_a = pose_a.cpu().numpy().squeeze()
        img_b = pose_b.cpu().numpy().squeeze()
        
        # Create difference visualization
        difference = cv2.addWeighted(img_a, blend_factor, img_b, 1-blend_factor, 0)
        
        # Convert back to tensor
        difference_tensor = torch.from_numpy(difference).unsqueeze(0)
        return (difference_tensor,)

class PoseDatabaseVisualizerNode:
    def show_database(self):
        # Implementation to display stored poses
        pass

NODE_CLASS_MAPPINGS = {
    "KeypointsToPoseNode": KeypointsToPoseNode,
    "KeypointsInputNode": KeypointsInputNode,
    "PoseEstimatorNode": PoseEstimatorNode,
    "KeypointComparatorNode": KeypointComparatorNode,
    "PoseDatabase": PoseDatabaseNode,
    "PoseDifference": PoseDifferenceNode,
    "PoseDatabaseVisualizer": PoseDatabaseVisualizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeypointsToPoseNode": "Pose Renderer (Stick Figure)",
    "KeypointsInputNode": "Pose Keypoints Input",
    "PoseEstimatorNode": "Pose Estimator",
    "KeypointComparatorNode": "Keypoint Comparison",
    "PoseDatabase": "Pose Database",
    "PoseDifference": "Pose Difference Visualizer",
    "PoseDatabaseVisualizer": "Pose Database Visualizer"
}
