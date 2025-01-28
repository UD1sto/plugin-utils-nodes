import torch
import numpy as np
import cv2
from PIL import Image
from .image_nodes import ImageNodeBase


def image_to_simhash(image_tensor):
    # Convert tensor to PIL Image
    image = Image.fromarray(np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    # Convert image to grayscale and resize for consistent hashing
    image = image.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
    
    # Calculate average pixel value
    avg = sum(image.getdata()) / 64
    
    # Generate hash (1 for pixels > average, 0 otherwise)
    bits = ''.join(['1' if (pixel > avg) else '0' for pixel in image.getdata()])
    
    return int(bits, 2)

def image_to_phash(image_tensor, hash_size=16, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    
    # Convert tensor to numpy array
    image_np = 255. * image_tensor.cpu().numpy().squeeze()
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    
    # Convert to grayscale and resize
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Apply DCT and focus on upper-left 8x8/16x16 frequencies
    dct = cv2.dct(cv2.dct(resized.astype(np.float32)))[:hash_size,:hash_size]
    
    # Calculate median and create binary hash
    median = np.median(dct)
    hash_array = (dct > median).flatten()
    return int(''.join(['1' if bit else '0' for bit in hash_array]), 2)

# Add this additional hash method
def image_to_dhash(image_tensor, hash_size=8):
    image = Image.fromarray(np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    
    pixels = list(image.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            difference.append(pixels[row*hash_size + col] > pixels[row*hash_size + col + 1])
    
    return int(''.join(['1' if bit else '0' for bit in difference]), 2)

def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

# apply_tooltips
class SimHashCompareNode(ImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "method": (["phash", "simhash", "dhash"], {"default": "phash"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("hamming_distance",)
    FUNCTION = "compare_images"
    CATEGORY = "Image/Comparison"

    def compare_images(self, image_a, image_b, method="phash"):
        if method == "phash":
            hash_a = image_to_phash(image_a)
            hash_b = image_to_phash(image_b)
        elif method == "simhash":
            # Original method
            hash_a = image_to_simhash(image_a)
            hash_b = image_to_simhash(image_b)
        else:
            # Difference Hash
            hash_a = image_to_dhash(image_a)
            hash_b = image_to_dhash(image_b)
        distance = hamming_distance(hash_a, hash_b)
        return (distance,)

#apply_tooltips
class ImageSelectorNode(ImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "hamming_distance": ("INT", {"default": 0, "min": 0, "max": 64}),
                "threshold": ("INT", {"default": 5, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("selected_image", "update_trigger")
    FUNCTION = "select_image"
    CATEGORY = "Image/Comparison"

    def select_image(self, image_a, image_b, hamming_distance, threshold):
        should_update = hamming_distance > threshold
        return (image_a if should_update else image_b, should_update)

#apply_tooltips
class TemporalConsistencyNode(ImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_frame": ("IMAGE",),
                "previous_reference": ("IMAGE",),
                "initial_reference": ("IMAGE",),
                "hamming_threshold": ("INT", {"default": 12, "min": 0, "max": 64}),
                "blend_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("output_frame", "updated_reference")
    FUNCTION = "process_frame"
    CATEGORY = "Image/Temporal"

    def process_frame(self, new_frame, previous_reference, initial_reference, hamming_threshold, blend_strength):
        # First frame detection
        is_first_frame = torch.equal(previous_reference, initial_reference)
        
        # Calculate perceptual hash difference
        current_hash = image_to_phash(new_frame)
        reference_hash = image_to_phash(previous_reference if not is_first_frame else initial_reference)
        distance = hamming_distance(current_hash, reference_hash)

        # Force update on first frame or significant change
        should_update = is_first_frame or (distance > hamming_threshold)

        # Create smooth transition
        blended_frame = self._blend_frames(
            new_frame if should_update else previous_reference,
            previous_reference,
            blend_strength
        )

        # Update reference logic
        new_reference = new_frame if should_update else previous_reference
        
        return (blended_frame, new_reference)

    def _blend_frames(self, current, previous, strength):
        return previous * (1 - strength) + current * strength

#apply_tooltips
class ImageReferenceUpdateNode(ImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "new_image": ("IMAGE",),
                "update_trigger": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("updated_reference",)
    FUNCTION = "update_reference"
    CATEGORY = "Image/Temporal"

    def update_reference(self, reference_image, new_image, update_trigger):
        return (new_image if update_trigger else reference_image,)

#apply_tooltips
class FrameBlendNode(ImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frame": ("IMAGE",),
                "previous_frame": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["linear", "exponential"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_frames"
    CATEGORY = "Image/Temporal"

    def blend_frames(self, current_frame, previous_frame, blend_factor, blend_mode):
        if blend_mode == "exponential":
            blend_factor = blend_factor ** 2
            
        blended = previous_frame * (1 - blend_factor) + current_frame * blend_factor
        return (blended,)