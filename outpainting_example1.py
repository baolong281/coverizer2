# %%
# an example script of how to do outpainting with the diffusers inpainting pipeline
# this is basically just the example from
# https://huggingface.co/runwayml/stable-diffusion-inpainting
#%
from diffusers import StableDiffusionInpaintPipeline

from PIL import Image
import numpy as np
import torch

from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# load the image, extract the mask
rgba = Image.open('primed_image_with_alpha_channel.png')
mask_image = Image.fromarray(np.array(rgba)[:, :, 3] == 0)

# run the pipeline
prompt = "Face of a yellow cat, high resolution, sitting on a park bench."
# image and mask_image should be PIL images.
# The mask structure is white for outpainting and black for keeping as is
image = pipe(
    prompt=prompt,
    image=rgba,
    mask_image=mask_image,
).images[0]
image

# %%
# the vae does lossy encoding, we could get better quality if we pasted the original image into our result.
# this may yield visible edges
