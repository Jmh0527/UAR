from typing import List
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score


class Validator(nn.Module):
    def __init__(self, model, dataloader, checkpoint_path):
        """
        Args:
            model: The neural network model for validation.
            dataloder: User-defined dataloader.
            checkpoint_path: Dictionary mapping dataset names to checkpoint file paths.
        """

        super(Validator, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._eval_hook = None # Preprocess before eval

    @property
    def eval_hook(self):
        return self._eval_hook

    @eval_hook.setter
    def eval_hook(self, hook):
        self._eval_hook = hook

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

        if self._eval_hook:
            self._eval_hook()
        
        state_dict = torch.load(self.checkpoint_path, map_location='cpu')
        if 'model' in state_dict.keys():
            self.model.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        acc, ap, r_acc, f_acc, y_true, y_pred = self.validate()

        return acc, ap, r_acc, f_acc, y_true, y_pred

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

        print('****************************************************************')
        print(f'predict real image number:{(y_pred <= 0.5).sum()}')
        print(f'predict fake image number:{(y_pred > 0.5).sum()}')
        print('****************************************************************')
        r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] >= 0.5)
        f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] >= 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)

        return acc, ap, r_acc, f_acc, y_true, y_pred
