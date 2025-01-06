import argparse
import logging
from pathlib import Path

import torch
from torchvision import transforms
import numpy as np

from networks import NetworkRegistry
from preprocess import TransformRegistry


def infer_single_image(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load the model
    if args.model not in NetworkRegistry.list_registered():
        raise ValueError(f"Model {args.model} is not registered in NetworkRegistry.")
    model = NetworkRegistry[args.model]()  # Instantiate the model

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load the transform
    if args.transform in TransformRegistry.list_registered():
        transform = TransformRegistry[args.transform]()
    else:
        transform = transforms.ToTensor()  # Default to tensor conversion

    # Load the image
    img_path = Path(args.image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    # Preprocess the image
    image = np.load(img_path) if args.datatype == 'npy' else transforms.ToPILImage()(np.array(img_path))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)

    logger.info(f"Fake possibility for image {args.image_path}: {output.item()}")
    return output.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AIMClassifier', help="Defined in the NetworkRegistry: AIMClassifier or PatchCraft")
    parser.add_argument('--datatype', type=str, default='npy', help="Specify the input data type. Options: 'npy' for AIMClassifier or 'image' for PatchCraft.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image file.")
    parser.add_argument('--transform', type=str, default=None, help="Defined TransformRegistry: HighPassFilter or Patch.")

    args = parser.parse_args()
    infer_single_image(args)
    
# python inference.py --checkpoint /home/data2/jmh/checkpoints/AIMClassifier/epoch_0_model.pth --image_path /home/data2/jmh/demo/demo_aimnpy/1_fake/ILSVRC2012_test_00001592.npy