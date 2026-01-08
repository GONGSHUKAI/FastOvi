"""
FastAPI Server for FastOvi Video Generation
"""
import base64
import io
import logging
import math
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchvision.transforms.functional as TF
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, Field

from pipeline import OviFewstepInferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model
app = FastAPI(title="FastOvi Video Generation API", version="1.0.0")
pipeline: Optional[OviFewstepInferencePipeline] = None
config: Optional[dict] = None


class FastOviRequest(BaseModel):
    """Request model for video generation"""
    prompt: str = Field(..., description="Text prompt for video generation")
    image: Optional[str] = Field(None, description="Base64 encoded reference image (optional)")
    seed: int = Field(42, description="Random seed for reproducibility")
    video_guidance_scale: Optional[float] = Field(None, description="Video CFG scale (default from config)")
    audio_guidance_scale: Optional[float] = Field(None, description="Audio CFG scale (default from config)")
    fps: int = Field(24, description="Output video FPS")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A man speaking into a microphone with text captions",
                "image": "base64_encoded_image_string_here",
                "seed": 42,
                "video_guidance_scale": 2.0,
                "audio_guidance_scale": 1.5,
                "fps": 24
            }
        }


def merge_audio_video(video_path: str, audio_path: str, output_path: str, fps: int = 24) -> bool:
    """Merge audio and video using ffmpeg"""
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-shortest',
        '-r', str(fps),
        output_path,
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Successfully merged audio and video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg.")
        return False


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int):
    """Save audio waveform to file"""
    waveform_np = waveform.squeeze().cpu().float().numpy()
    sf.write(path, waveform_np, sample_rate)


def load_and_process_image(image_base64: Optional[str], config) -> tuple:
    """
    Load and process reference image
    Returns: (wan22_image_latent, target_w, target_h)
    """
    if not image_base64:
        raise ValueError("image field with base64 encoded image is required")

    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Calculate target size
    orig_w, orig_h = img.size
    max_pixels = 704 * 1280

    if orig_w * orig_h > max_pixels:
        aspect_ratio = orig_w / orig_h
        target_h = int(math.sqrt(max_pixels / aspect_ratio))
        target_w = int(target_h * aspect_ratio)
    else:
        target_w, target_h = orig_w, orig_h

    # Round to multiple of 32
    target_w = (target_w // 32) * 32
    target_h = (target_h // 32) * 32

    logger.info(f"Input image size: ({orig_w}, {orig_h}), resized to: ({target_w}, {target_h})")

    # Process image
    img_resized = img.resize((target_w, target_h), Image.LANCZOS)
    img_tensor = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).unsqueeze(1).to("cuda", dtype=torch.bfloat16)
    wan22_image_latent = pipeline.vae.encode_video(img_tensor.unsqueeze(0))

    return wan22_image_latent, target_w, target_h


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global pipeline, config

    # Check FFmpeg
    if not shutil.which('ffmpeg'):
        logger.error("FFmpeg not found. Please install: sudo apt update && sudo apt install ffmpeg")
        raise RuntimeError("FFmpeg is required but not found")

    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "configs/ovi_smallcfg.yaml")
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "/root/dq/OviDMD/model_ema.pt")

    logger.info(f"Loading configuration from: {config_path}")
    config = OmegaConf.load(config_path)

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize pipeline
    logger.info("Initializing inference pipeline...")
    pipeline = OviFewstepInferencePipeline(config)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
        logger.info("Loaded 'generator_ema' weights")
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
        logger.info("Loaded 'generator' weights")
    else:
        logger.info("Loaded weights directly from checkpoint")

    # Clean state dict keys
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned_state_dict[new_key] = value

    # Load into model
    missing, unexpected = pipeline.generator.model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Move to GPU
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16).eval()

    logger.info("Model loaded successfully! Ready to generate videos.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pipeline
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()
    logger.info("Server shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "FastOvi Video Generation API is running",
        "model_loaded": pipeline is not None
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if pipeline is not None else "not_ready",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/generate")
async def generate_video(request: FastOviRequest):
    """
    Generate video from text prompt and reference image

    Returns video as bytes (application/octet-stream)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    logger.info(f"Received generation request: prompt='{request.prompt[:80]}...', seed={request.seed}")

    # Use temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Process reference image
            wan22_image_latent, target_w, target_h = load_and_process_image(
                request.image,
                config
            )

            # Generate noise
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
            video_noise = torch.randn(
                (1, (config.video_num_frames - 1) // 4 + 1, 48, target_h // 16, target_w // 16),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16
            )

            audio_latent_len, audio_latent_dim = 157, 20
            audio_noise = torch.randn(
                (1, audio_latent_len, audio_latent_dim),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16
            )

            # Get guidance scales
            video_guidance_scale = request.video_guidance_scale or config.video_guidance_scale
            audio_guidance_scale = request.audio_guidance_scale or config.audio_guidance_scale

            logger.info("Starting inference...")

            # Run inference
            video_out, audio_out = pipeline.inference(
                noise_video=video_noise,
                noise_audio=audio_noise,
                text_prompts=[request.prompt],
                wan22_image_latent=wan22_image_latent,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                video_negative_prompt=config.video_negative_prompt,
                audio_negative_prompt=config.audio_negative_prompt,
            )

            logger.info("Inference complete. Processing outputs...")

            # Save video and audio to temporary files
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
            final_video_path = os.path.join(temp_dir, "final_video.mp4")

            # Export video
            video_np = video_out[0].permute(0, 2, 3, 1).cpu().float().numpy()
            export_to_video(video_np, temp_video_path, fps=request.fps)

            # Save audio
            save_audio(audio_out, temp_audio_path, sample_rate=config.audio_sample_rate)

            # Merge audio and video
            merge_success = merge_audio_video(temp_video_path, temp_audio_path, final_video_path, fps=request.fps)

            if not merge_success:
                raise RuntimeError("Failed to merge audio and video")

            # Read final video as bytes
            with open(final_video_path, "rb") as f:
                video_bytes = f.read()

            logger.info(f"Video generation successful. Size: {len(video_bytes) / 1024 / 1024:.2f} MB")

            # Return video as bytes directly
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f'attachment; filename="video_{request.seed}.mp4"'
                }
            )

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting FastOvi API server on {host}:{port}")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
