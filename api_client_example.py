
"""
Example client for FastOvi API
"""
import base64
import requests
from pathlib import Path


def generate_video(
    api_url: str,
    prompt: str,
    image_path: str,
    seed: int = 42,
    output_path: str = "output.mp4"
):
    """Generate video using base64 encoded image"""

    # Read and encode image to base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "prompt": prompt,
        "image": image_base64,
        "seed": seed,
        "video_guidance_scale": 2.0,
        "audio_guidance_scale": 1.5,
        "fps": 24
    }

    print(f"Sending request to {api_url}/generate...")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Image: {image_path}")
    print(f"Image size (base64): {len(image_base64)} chars")
    print(f"Seed: {seed}")

    response = requests.post(f"{api_url}/generate", json=payload)

    if response.status_code == 200:
        # Response is directly bytes (video/mp4)
        video_bytes = response.content

        # Save to file
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        print(f"\n✅ Video saved to: {output_path}")
        print(f"Video size: {len(video_bytes) / 1024 / 1024:.2f} MB")

        return output_path
    else:
        print(f"\n❌ Error {response.status_code}: {response.text}")
        return None


def check_health(api_url: str):
    """Check API health status"""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"API Status: {health['status']}")
            print(f"CUDA Available: {health['cuda_available']}")
            print(f"Device Count: {health['device_count']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to API: {e}")
        return False


if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"

    # Check health first
    print("Checking API health...")
    if not check_health(API_URL):
        print("API is not ready. Please start the server first.")
        exit(1)

    print("\n" + "="*60)

    # Generate video with base64 image
    print("\nGenerating video...")
    generate_video(
        api_url=API_URL,
        prompt="A bearded man wearing large dark sunglasses and a blue patterned cardigan sits in a studio, actively speaking into a large, suspended microphone. He has headphones on and gestures with his hands, displaying rings on his fingers. Behind him, a wall is covered with red, textured sound-dampening foam on the left, and a white banner on the right features the \"CHOICE FM\" logo and various social media handles like \"@ilovechoicefm\" with \"RALEIGH\" below it. The man intently addresses the microphone, articulating, <S>is talent. It's all about authenticity. You gotta be who you really are, especially if you're working<E>. He leans forward slightly as he speaks, maintaining a serious expression behind his sunglasses.. <AUDCAP>Clear male voice speaking into a microphone, a low background hum.<ENDAUDCAP>",
        image_path="/cpfs01/gongshukai/step_distillation/examples/image_ovi2/5.png",
        seed=42,
        output_path="output_example1.mp4"
    )

    print("\n" + "="*60)
    print("\n✨ Generation completed!")
