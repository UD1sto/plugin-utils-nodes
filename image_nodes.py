import torch

class ImageNodeBase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "Image"
    FUNCTION = "process"

    def process(self, image, mask=None):
        if mask is not None:
            # Basic masking operation
            image = image * mask.unsqueeze(-1)
        return (image,) 