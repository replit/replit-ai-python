import os
import requests
import aiohttp
from .model import Model
from .structs import ImageGenerationModelResponse, LoraWeight, ImageSizeInput
from typing import Optional, Any, Dict, Literal

DEFAULT_LORA_URL = "https://110602490-lora.gateway.alpha.fal.ai"


Scheduler = Literal[
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "Euler",
    "Euler A",
]


class ImageGenerationModel(Model):
    """Generates images on fal infrastructure."""

    def __init__(
        self,
        model_url: Optional[str] = None,
        fal_key_id: Optional[str] = None,
        fal_key_secret: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Initializes a ImageGenerationModel instance.

        Args:
          model_url (Optional[str]): Url to serving ImageGeneration Lora model
          fal_key_id (Optional[str]): Fal credentials key id
          fal_key_secret (Optional[str]): Fal credentials key secret
          **kwargs (Dict[str, Any]): Additional keyword arguments.
        """
        self.model_url = model_url if model_url else DEFAULT_LORA_URL
        self.fal_key_id = fal_key_id if fal_key_id else os.environ.get("FAL_KEY_ID")
        self.fal_key_secret = (
            fal_key_secret if fal_key_secret else os.environ.get("FAL_KEY_SECRET")
        )

        if not self.fal_key_id or not self.fal_key_secret:
            raise ValueError(
                "Both fal_key_id and fal_key_secret must be provided or set as environment variables."
            )

    async def async_generate(
        self,
        prompt: str,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        negative_prompt: str = "",
        loras: list[LoraWeight] = [],
        seed: Optional[int] = None,
        image_size: ImageSizeInput = "square_hd",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        clip_skip: int = 0,
        model_architecture: Optional[Literal["sd", "sdxl"]] = None,
        scheduler: Optional[Scheduler] = None,
        image_format: Literal["jpeg", "png"] = "png",
        num_images: int = 1,
    ) -> ImageGenerationModelResponse:
        """
        Asynchronously generate an image from a text prompt.

        Parameters:
            prompt (str): The prompt to use for generating the image
            model_name (str): URL or HuggingFace ID of the base model to generate the image
            negative_prompt (str): Prompt for details that you don't want in the image
            loras (list[LoraWeight]): The LoRAs to use for the image generation
            seed (int): Same seed and prompt given to a model will result in same image
            image_size (ImageSizeInput): The size of the generated image
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale (CFG)
            clip_skip (int): Skips part of the image generation process
            model_architecture (str): The architecture of the model to use, "sd" or "sdxl"
            scheduler (str): Scheduler / sampler to use for the image denoising process
            image_format (str): The format of the generated image, "png" or "jpeg"
            num_images (int): Number of images to generate in one request

        Returns:
            ImageGenerationModelResponse
        """
        url = f"{self.model_url}"
        headers = {"Authorization": f"Basic {self.fal_key_id}:{self.fal_key_secret}"}
        params = {
            "prompt": prompt,
            "model_name": model_name,
            "negative_prompt": negative_prompt,
            "loras": loras,
            "seed": seed,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "clip_skip": clip_skip,
            "model_architecture": model_architecture,
            "scheduler": scheduler,
            "image_format": image_format,
            "num_images": num_images,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=params) as response:
                await self._check_aresponse(response)
                return ImageGenerationModelResponse(**await response.json())

    def generate(
        self,
        prompt: str,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        negative_prompt: str = "",
        loras: list[LoraWeight] = [],
        seed: Optional[int] = None,
        image_size: ImageSizeInput = "square_hd",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        clip_skip: int = 0,
        model_architecture: Optional[Literal["sd", "sdxl"]] = None,
        scheduler: Optional[Scheduler] = None,
        image_format: Literal["jpeg", "png"] = "png",
        num_images: int = 1,
    ):
        """
        Generate an image from a text prompt.

        Parameters:
            prompt (str): The prompt to use for generating the image
            model_name (str): URL or HuggingFace ID of the base model to generate the image
            negative_prompt (str): Prompt for details that you don't want in the image
            loras (list[LoraWeight]): The LoRAs to use for the image generation
            seed (int): Same seed and prompt given to a model will result in same image
            image_size (ImageSizeInput): The size of the generated image
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale (CFG)
            clip_skip (int): Skips part of the image generation process
            model_architecture (str): The architecture of the model to use, "sd" or "sdxl"
            scheduler (str): Scheduler / sampler to use for the image denoising process
            image_format (str): The format of the generated image, "png" or "jpeg"
            num_images (int): Number of images to generate in one request

        Returns:
            ImageGenerationModelResponse
        """
        url = f"{self.model_url}"
        headers = {"Authorization": f"Basic {self.fal_key_id}:{self.fal_key_secret}"}
        response = requests.post(
            url,
            headers=headers,
            json={
                "prompt": prompt,
                "model_name": model_name,
                "negative_prompt": negative_prompt,
                "loras": loras,
                "seed": seed,
                "image_size": image_size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "clip_skip": clip_skip,
                "model_architecture": model_architecture,
                "scheduler": scheduler,
                "image_format": image_format,
                "num_images": num_images,
            },
        )
        self._check_response(response)
        return ImageGenerationModelResponse(**response.json())
