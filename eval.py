import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

from networks import NetworkRegistry
from dataset import BaseDataset
from preprocess import TransformRegistry
from eval_engine import Validator

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load the model
    if args.model not in NetworkRegistry.list_registered():
        raise ValueError(f"Model {args.model} is not registered in NetworkRegistry.")
    model = NetworkRegistry[args.model]()  # Instantiate the model

    # Load transform
    if args.transform in TransformRegistry.list_registered():
        transform = TransformRegistry[args.transform]()
    else:
        transform = None

    # Load dataset and dataloader
    val_dataloaders = []
    for val in args.validation_sets:
        dataset = BaseDataset(img_paths=Path(args.dataroot)/Path(val), transform=transform, data_type=args.datatype)
        val_dataloaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))

    def write_or_log(line, output_path):
        if output_path:
            with open(output_path, 'a') as f:
                f.write(line+'\n')
        else:
            logger.info(line)

    # Validator
    ACC = []
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    for idx, dataloader in enumerate(val_dataloaders):
        validator = Validator(model, dataloader, args.checkpoint)
        acc, ap, r_acc, f_acc, y_true, y_pred = validator.eval()
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        # Convert probabilities to binary predictions using a threshold of 0.5
        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Calculate TP, FP, TN, FN
        TP = ((y_true == 1) & (y_pred_binary == 1)).sum()
        FP = ((y_true == 0) & (y_pred_binary == 1)).sum()
        TN = ((y_true == 0) & (y_pred_binary == 0)).sum()
        FN = ((y_true == 1) & (y_pred_binary == 0)).sum()

        # Calculate Precision and Recall
        precision_value = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_value = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate F1 score
        f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0

        # Output the counts and F1 score
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"F1 Score: {f1_score:.4f}")
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        # Save the plot to the specified path
        name = args.validation_sets[idx]
        plt.savefig(f'./PR_curve/{name}.png')

        result_line = f"{args.validation_sets[idx]} acc: {acc:.4f}, ap: {ap:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}, F1 Score: {f1_score:.4f}"
        ACC.append(acc) 
        write_or_log(result_line, args.output)

    average_acc = sum(ACC) / len(ACC)
    average_line = f"average acc: {average_acc:.4f}\n"
    write_or_log(average_line, args.output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AIMClassifier', help="Defined in the NetworkRegistry: AIMClassifier or PatchCraft")
    parser.add_argument('--datatype', type=str, default='npy', help="Specify the input data type. Options: 'npy' for AIMClassifier or 'image' for PatchCraft.")
    parser.add_argument('--dataroot', type=str, required=True, help="Path to the validation dataset root.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Directory containing model checkpoints.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for validation.")
    parser.add_argument('--transform', type=str, default=None, help="Defined TransformRegistry: HighPassFilter or Patch.")
    parser.add_argument(
        '--validation_sets',
        nargs='+',
        help="List of validation set names, if you don't have child dir, then give the value of ''. ",
        default=[
            'progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
            'stylegan2', 'whichfaceisreal', 'ADM', 'Glide', 'Midjourney',
            'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong', 'DALLE2'
        ]
    )
    parser.add_argument('--output', type=str, default=None, help="Output file for validation results.")
    
    args = parser.parse_args()
    main(args)