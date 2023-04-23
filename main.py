
from PIL import Image, ImageOps
import requests
from io import BytesIO

import tempfile
import os
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage


def images_to_video(images: list[Image.Image], output_path: str, fps: int = 30) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save PIL images to temporary directory with sequential naming
        for idx, image in enumerate(images):
            image.save(os.path.join(temp_dir, f"frame_{idx:04d}.png"))

        # Call ffmpeg using os.system()
        input_pattern = os.path.join(temp_dir, "frame_%04d.png")
        crf = 1  # Lower values result in better quality, range: 0-51 (18-28 recommended)
        os.system(f"ffmpeg -framerate {fps} -i {input_pattern} -y -c:v mpeg4 -vtag mp4v -q:v {crf} -pix_fmt yuv420p {output_path}")


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


def shift_noise(noise: torch.Tensor, shift_pixels: int=-1) -> torch.Tensor:
    if not isinstance(shift_pixels, int):
        raise ValueError("shift_pixels must be an integer.")

    shifted_noise = noise.clone()

    if shift_pixels > 0:
        # Shift noise to the right
        shifted_noise[..., shift_pixels:] = noise[..., :-shift_pixels]
        shifted_noise[..., :shift_pixels] = torch.randn_like(noise[..., :shift_pixels])
    elif shift_pixels < 0:
        # Shift noise to the left
        shifted_noise[..., :shift_pixels] = noise[..., -shift_pixels:]
        shifted_noise[..., shift_pixels:] = torch.randn_like(noise[..., shift_pixels:])
    # If shift_pixels == 0, return the original noise

    return shifted_noise


def main():
    url = "https://pbs.twimg.com/media/E6_U0p4WEAkYPx_.jpg:large"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content))
    w, h = init_image.size
    crop_size = 1024
    init_image = init_image.crop(((w - crop_size) / 2, (h - crop_size) / 2, (w-crop_size)/2 + crop_size, (h-crop_size)/2 + crop_size))
    init_image = init_image.resize((512, 512))
    init_image.save("init_img.png")

    N = 50
    init_images = shift_image(init_image, N)
    #N = 10
    #torch.manual_seed(2)
    #image_tensor = ToTensor()(init_image)
    #shifted_tensors = [image_tensor]
    #for _ in range(N):
    #    shifted_tensors.append(shift_noise(shifted_tensors[-1], 1))

    #shifted_images = [ToPILImage()(tensor) for tensor in shifted_tensors[1:]]   
    images_to_video(init_images, f"init_{N}.mp4", fps=3)


    model_id = "stabilityai/stable-diffusion-2-1"
    euler = EulerAncestralDiscreteScheduler.from_config(model_id, subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=euler)
    pipe.disable_xformers_memory_efficient_attention()
    #pipe.scheduler = lms = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = "portrait photography of a woman looking at the camera, symmetrical eyes, blue eyes, piercing eyes, soft light, pale skin, flushed cheeks, birth mark, spots, freckles, pock marks, fine lines, intimate portrait composition, high detail, engaging, friendly, characterful, a hyper-realistic close-up portrait, portrait photography, essay, portrait photo, photorealism, leica 28mm, DSLRs, sharp, artgerm, post-processing highly detailed, pock marks, creases, imperfections, spots, dry skin, pores, freckles, natural"
    prompt_neg = "fake, avatar, foggy eyes, bad eyes, asymetrical eyes, shiny, airbrushed, make up, blurry, perfect, smooth, 3D graphics, video games, unreal engine, CGI, out of focus, muted colors, dripping paint, 2 heads, 2 faces, cropped image, out of frame, deformed hands, twisted fingers, double image, malformed hands, multiple heads, xtra limb, poorly drawn hands, missing limb, disfigured, cut off, low res, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, poorly drawn, mutilated, mangled, extra fingers, duplicate artifacts, missing arms, mutated hands, mutilated hands, cloned face, malformed, text, logo, wordmark, writing, heading, signature, long neck"
    device = "cuda"

    outs = []
    for i, img in enumerate(init_images):
        generator = torch.Generator(device=device).manual_seed(2)

        def transform(noise):
            return shift_noise(noise, int((-N + i) * 64/512))
        
        image = pipe(
            prompt="A photo of a woman looking at the camera",
            negative_prompt="",
            image=init_image,
            strength=0.3,
            guidance_scale=7,
            generator=generator,
            noise_transform=shift_noise,
            img_id=i,
        ).images[0]
        outs.append(image)
    
    images_to_video(outs, f"out_{N}_bilinear.mp4", fps=30)
    
    

if __name__ == "__main__":
    main()