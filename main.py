
from PIL import Image, ImageOps
import requests
from io import BytesIO

import tempfile
import os
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
import numpy as np


def images_to_video(images: list[Image.Image], output_path: str, fps: int = 30) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save PIL images to temporary directory with sequential naming
        for idx, image in enumerate(images):
            image.save(os.path.join(temp_dir, f"frame_{idx:04d}.png"))

        # Call ffmpeg using os.system()
        input_pattern = os.path.join(temp_dir, "frame_%04d.png")
        os.system(f"ffmpeg -framerate {fps} -y -i {input_pattern} -pix_fmt yuv420p -crf 1 {output_path}")


def shift_image(image: Image.Image, shift_pixels: int) -> list[Image.Image]:
    shifted_images = []
    width, height = image.size

    for shift in range(-shift_pixels, shift_pixels + 1):
        if shift < 0:
            # Shift the image to the left
            left_expanded = ImageOps.expand(image, (-shift, 0, 0, 0))
            left_shifted = left_expanded.crop((0, 0, width, height))
            shifted_images.append(left_shifted)
        elif shift > 0:
            # Shift the image to the right
            right_expanded = ImageOps.expand(image, (0, 0, shift, 0))
            right_shifted = right_expanded.crop((shift, 0, width + shift, height))
            shifted_images.append(right_shifted)
        else:
            # No shift
            shifted_images.append(image.copy())

    return shifted_images


def main():
    url = "https://pbs.twimg.com/media/E6_U0p4WEAkYPx_.jpg:large"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content))
    w, h = init_image.size
    crop_size = 1024
    init_image = init_image.crop(((w - crop_size) / 2, (h - crop_size) / 2, (w-crop_size)/2 + crop_size, (h-crop_size)/2 + crop_size))
    init_image = init_image.resize((512, 512))
    init_image.save("init_img.png")

    model_id = "stabilityai/stable-diffusion-2-1"
    euler = EulerAncestralDiscreteScheduler.from_config(model_id, subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=euler)
    pipe.disable_xformers_memory_efficient_attention()
    #pipe.scheduler = lms = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = "portrait photography of a woman looking at the camera, symmetrical eyes, blue eyes, piercing eyes, soft light, pale skin, flushed cheeks, birth mark, spots, freckles, pock marks, fine lines, intimate portrait composition, high detail, engaging, friendly, characterful, a hyper-realistic close-up portrait, portrait photography, essay, portrait photo, photorealism, leica 28mm, DSLRs, sharp, artgerm, post-processing highly detailed, pock marks, creases, imperfections, spots, dry skin, pores, freckles, natural"
    prompt_neg = "fake, avatar, foggy eyes, bad eyes, asymetrical eyes, shiny, airbrushed, make up, blurry, perfect, smooth, 3D graphics, video games, unreal engine, CGI, out of focus, muted colors, dripping paint, 2 heads, 2 faces, cropped image, out of frame, deformed hands, twisted fingers, double image, malformed hands, multiple heads, xtra limb, poorly drawn hands, missing limb, disfigured, cut off, low res, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, poorly drawn, mutilated, mangled, extra fingers, duplicate artifacts, missing arms, mutated hands, mutilated hands, cloned face, malformed, text, logo, wordmark, writing, heading, signature, long neck"
    device = "cuda"

    init_images = shift_image(init_image, 100)
    
    images_to_video(init_images, "init_100.mp4", fps=30)
    outs = []
    for img in init_images:
        generator = torch.Generator(device=device).manual_seed(2)
        image = pipe(
            prompt="A photo of a woman looking at the camera",
            negative_prompt="",
            image=img,
            strength=0.3,
            guidance_scale=7,
            generator=generator
        ).images[0]
        outs.append(image)
    
    images_to_video(outs, "out_100.mp4", fps=30)
    
    

if __name__ == "__main__":
    main()