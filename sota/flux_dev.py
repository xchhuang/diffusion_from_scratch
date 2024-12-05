import torch
from diffusers import FluxPipeline
import os

# Steps to get permission:
# huggingface-cli login (with created access token)


output_folder = "results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.enable_model_cpu_offload() 

# prompt = "A cat holding a sign that says hello world"
# prompt = "A dog holding a sign that says hello world"
prompt = "A corgi holding a sign that says hello fluxdev"
# prompt = "an old rusted robot wearing pants and a jacket riding skis in a supermarket"

image = pipe(
    prompt,
    height=512, # 1024
    width=512,  # # 1024
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
save_file_name = '_'.join(prompt.split(' '))
image.save(f"{output_folder}/{save_file_name}.png")

