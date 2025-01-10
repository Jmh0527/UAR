import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torchvision.datasets import VisionDataset
from diffusers import AutoPipelineForImage2Image
from diffusers.models import VQModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from joblib.hashing import hash
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as tf
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import tqdm
import argparse

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


class ImageFolder(VisionDataset):
    """
    Dataset for reading images from a list of paths, directories, or a mixture of both.
    """

    def __init__(
        self,
        paths: Union[list[Path], Path],
        transform: Optional[Callable] = tf.Compose(
            [tf.ToImage(), tf.ToDtype(torch.float32, scale=True)]
        ),
        amount: Optional[int] = None,
    ) -> None:
        self.paths = [paths] if isinstance(paths, Path) else paths
        self.transform = transform
        self.amount = amount

        self.img_paths = []
        for path in self.paths:
            if path.is_dir():
                for file in sorted(path.iterdir()):
                    if file.suffix.lower() in IMG_EXTENSIONS:
                        self.img_paths.append(file)
                        if (
                            self.amount is not None
                            and len(self.img_paths) == self.amount
                        ):
                            break
            else:
                self.img_paths.append(path)

        if self.amount is not None and len(self.img_paths) < self.amount:
            raise ValueError("Number of images is less than 'amount'.")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[str, float]]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, str(self.img_paths[idx])

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Paths: {self.paths}")
        body.append(f"Transform: {repr(self.transform)}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


def safe_mkdir(directory: Path) -> None:
    """Ask before using an existing directory."""
    if directory.exists():
        response = input(
            f"Directory '{str(directory)}' exists, continue? (y/n) "
        ).lower()
        if response not in ["yes", "y"]:
            exit()
    directory.mkdir(parents=True, exist_ok=True)

def device() -> str:
    """Return 'cuda' if available, 'cpu' otherwise"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def reconstruction_image(
    ds: Dataset,
    repo_id: str,
    output_dir: Optional[Path] = None,
    seed: int = 1,
    batch_size: int = 1,
    num_workers: int = 1,
) -> None:
    
    safe_mkdir(output_dir)

    # set up pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()

    # extract AE
    if hasattr(pipe, "vae"):
        ae = pipe.vae
        if hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
    elif hasattr(pipe, "movq"):
        ae = pipe.movq
    ae.to(device())
    ae = torch.compile(ae)
    decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype

    # reconstruct
    generator = torch.Generator().manual_seed(seed)
    reconstruction_paths = []
    for images, paths in tqdm(
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers),
        desc=f"Reconstructing with {repo_id}.",
    ):
        # normalize
        images = images.to(device(), dtype=ae.dtype) * 2.0 - 1.0

        # encode
        try:
            latents = retrieve_latents(ae.encode(images), generator=generator)
            # decode
            if isinstance(ae, VQModel):
                reconstructions = ae.decode(
                    latents.to(decode_dtype), force_not_quantize=True, return_dict=False
                )[0]
            else:
                reconstructions = ae.decode(
                    latents.to(decode_dtype), return_dict=False
                )[0]

            # de-normalize
            reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)

            # save
            for reconstruction, path in zip(reconstructions, paths):
                reconstruction_path = output_dir / f"{Path(path).stem}.png"
                to_pil_image(reconstruction).save(reconstruction_path)
        except:
            with open('./reconstruct_error.txt', 'a') as f:
                f.write(paths[0] + '\n')
            continue
    print(f"Images saved to {output_dir}.")
    return reconstruction_paths

def main(args):
    ds = ImageFolder(Path(args.input_dir))
    reconstruction_image(
        ds,
        repo_id=Path(args.repo_id),
        output_dir=Path(args.output_dir),
        seed=1,
        batch_size=1,
        num_workers=1,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct real images for training UAR.")
    parser.add_argument("--repo_id", type=str, default="./stable-diffusion-v1-4")
    parser.add_argument("--input_dir", type=str, default="./image_data/0_real")
    parser.add_argument("--output_dir", type=str, default="./image_data/1_fake")
    
    args = parser.parse_args()
    main(args)