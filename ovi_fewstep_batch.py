# FILE: ovi_fewstep_batch.py (FINAL, WITH MERGING)
from pipeline import OviFewstepInferencePipeline
import argparse
import csv
import math
import os
import shutil
import subprocess
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
import soundfile as sf
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s] %(levelname)s: %(message)s"
)

def merge_audio_video(video_path, audio_path, output_path, fps=24):
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Runtime error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("❌ File not found error")
        return False


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int):
    waveform_np = waveform.squeeze().cpu().float().numpy()
    sf.write(path, waveform_np, sample_rate)

def process_one(pipe: OviFewstepInferencePipeline, prompt, image_path, seed, idx, config, args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"line_{idx:03d}_seed_{seed}"
    final_output_path = os.path.join(output_dir, f"{base_filename}_final.mp4")
    temp_video_path = os.path.join(output_dir, f"{base_filename}_temp_video.mp4")
    temp_audio_path = os.path.join(output_dir, f"{base_filename}_temp_audio.wav")

    target_w, target_h = config.video_h, config.video_w
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        max_pixels = 704 * 1280
        if orig_w * orig_h > max_pixels:
            aspect_ratio = orig_w / orig_h
            target_h = int(math.sqrt(max_pixels / aspect_ratio))
            target_w = int(target_h * aspect_ratio)
        else:
            target_w, target_h = orig_w, orig_h

        target_w = (target_w // 32) * 32
        target_h = (target_h // 32) * 32
        print(f"🖼️ Input image size: ({orig_w}, {orig_h}), resized to: ({target_w}, {target_h}) with the same aspect ratio")

        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        img_tensor = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).unsqueeze(1).to("cuda", dtype=torch.bfloat16)
        wan22_image_latent = pipe.vae.encode_video(img_tensor.unsqueeze(0))
    else:
        raise ValueError(f"TI2AV needs a reference frame but there is no {image_path}.")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    video_noise = torch.randn(
        (1, (config.video_num_frames - 1) // 4 + 1, 48, target_h // 16, target_w // 16), 
        generator=generator, 
        device="cuda", 
        dtype=torch.bfloat16
    )   # (1, lat_F, lat_C, lat_H, lat_W)
    audio_latent_len, audio_latent_dim = 157, 20
    audio_noise = torch.randn(
        (1, audio_latent_len, audio_latent_dim), generator=generator, device="cuda", dtype=torch.bfloat16
    )

    print("🚀 Generating Video...")
    video_out, audio_out = pipe.inference(
        noise_video=video_noise,
        noise_audio=audio_noise,
        text_prompts=[prompt],
        wan22_image_latent=wan22_image_latent,
        video_guidance_scale=config.video_guidance_scale,
        audio_guidance_scale=config.audio_guidance_scale,
        video_negative_prompt=config.video_negative_prompt,
        audio_negative_prompt=config.audio_negative_prompt,
    )

    video_np = video_out[0].permute(0, 2, 3, 1).cpu().float().numpy()
    export_to_video(video_np, temp_video_path, fps=24)
    save_audio(audio_out, temp_audio_path, sample_rate=config.audio_sample_rate)
    merge_success = merge_audio_video(temp_video_path, temp_audio_path, final_output_path)
    if merge_success:
        os.remove(temp_video_path)
        os.remove(temp_audio_path)

def main():
    if not shutil.which('ffmpeg'):
        print("You need to install FFMpeg")
        print("sudo apt update && sudo apt install ffmpeg")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True, help="csv path for TI2AV")
    parser.add_argument("--h", type=int, default=704, help="default video height")
    parser.add_argument("--w", type=int, default=1280, help="default video width")
    parser.add_argument("--output_dir", type=str, default="outputs", help="generated content directory")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = OviFewstepInferencePipeline(config)

    print(f"Loading distilled checkpoint from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    
    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
        print("Loaded 'generator_ema' weights.")
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
        print("Loaded 'generator' weights.")
    else:
        print("Loaded weights directly from checkpoint.")
    # if "generator" in state_dict:
    #     state_dict = state_dict["generator"]
    #     print("Loaded 'generator' weights.")
    # else:
    #     print("Loaded weights directly from checkpoint.")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned_state_dict[new_key] = value

    missing, unexpected = pipe.generator.model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"⚠️ Missing keys: {missing}")
    print(f"⚠️ Unexpected keys: {unexpected}")
    
    pipe = pipe.to(device="cuda", dtype=torch.bfloat16).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt = row["prompt"]
            image_path = row.get("image")
            seed = int(row.get("seed", 42))
            
            print("\n" + "="*50)
            print(f"🎬 Processing prompt {idx+1}: {prompt[:80]}..., seed: {seed}")
            process_one(pipe, prompt, image_path, seed, idx + 1, config, args)
            print("="*50)

if __name__ == "__main__":
    main()