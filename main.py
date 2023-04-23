
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


def main():
    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = lms = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    init_image = ...
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(69)

    image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
        
    image.save("astronaut_rides_horse.png")

if __name__ == "__main__":
    main()