import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score


class Validator(nn.Module):
    validation_sets = [
        'progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'stylegan2', 'whichfaceisreal', 'ADM', 'Glide', 'Midjourney',
        'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong', 'DALLE2'
    ]

    def __init__(self, model, dataloader, checkpoint_paths):
        """
        Args:
            model: The neural network model for validation.
            dataloder: User-defined dataloader.
            checkpoint_paths: Dictionary mapping dataset names to checkpoint file paths.
        """

        super(Validator, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.checkpoint_paths = checkpoint_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval(self) -> List[List]:
        """
        Evaluates the model across all validation sets.
        
        Returns:
            results: List of lists, where each inner list contains:
                - float: Overall accuracy.
                - float: Average precision (AP).
                - float: Accuracy for real samples (r_acc).
                - float: Accuracy for fake samples (f_acc).
        """

        results = []
        for val in self.validation_sets:

            # Load the model checkpoint
            state_dict = torch.load(self.checkpoint_paths[val], map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=True)
            self.model.to(self.device)
            self.model.eval()

            acc, ap, r_acc, f_acc, _, _ = self.validate()
            results.append([val, acc, ap, r_acc, f_acc])
            print(f"({val}) acc: {acc}; ap: {ap}; r_acc: {r_acc}; f_acc: {f_acc}")

        return results

    def validate(self):
        """
        Validates the model on a single dataloader.
        """

        y_true, y_pred = [], []

        for batch_idx, (img, label) in enumerate(self.dataloader):
            print(f"Processing batch {batch_idx}/{len(self.dataloader)}", end='\r')
            in_tensors = img.cuda()

            preds = self.model(in_tensors).sigmoid().flatten().tolist()
            y_pred.extend(preds)
            y_true.extend(label.flatten().tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
        f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)

        return acc, ap, r_acc, f_acc, y_true, y_pred
