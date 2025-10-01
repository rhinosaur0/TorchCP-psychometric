# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.predictor.split import SplitPredictor
from torchcp.classification.utils import ConfCalibrator
from torchcp.classification.utils.metrics import Metrics


class ClassConditionalPredictor(SplitPredictor):
    """
    Method: Class-conditional conformal prediction
    Paper: Conditional validity of inductive conformal predictors (Vovk et al., 2012)
    Link: https://proceedings.mlr.press/v25/vovk12.html
    
        
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        alpha (float, optional): The significance level. Default is 0.1.
        device (torch.device, optional): The device on which the model is located. Default is None.

    Attributes:
        q_hat (torch.Tensor): The calibrated threshold for each class.
    """

    def __init__(self, score_function, model=None, temperature=1, alpha=0.1, device=None):

        super(ClassConditionalPredictor, self).__init__(score_function, model, temperature, alpha, device)
        self.q_hat = None

    def calculate_threshold(self, logits, labels, alpha=None):
        """
        Calculate the class-wise conformal prediction thresholds.

        Args:
            logits (torch.Tensor): The logits output from the model.
            labels (torch.Tensor): The ground truth labels.
            alpha (float): The significance level. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        alpha = torch.tensor(alpha, device=self._device)
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        # Count the number of classes
        num_classes = logits.shape[1]
        self.q_hat = torch.zeros(num_classes, device=self._device)
        scores = self.score_function(logits, labels)
        for label in range(num_classes):
            temp_scores = scores[labels == label]
            self.q_hat[label] = self._calculate_conformal_value(temp_scores, alpha)

class XGBoostClassConditionalPredictor(ClassConditionalPredictor):
    def __init__(self, score_function, model, temperature=1, alpha=0.1, device=None):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        self.score_function = score_function
        self._model = model # Keep _model as the XGBoost model

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")
        self.alpha = alpha

        # For XGBoost, device is less relevant for the model itself,
        # but we keep it for tensor operations later.
        if device is not None:
            self._device = torch.device(device)
        else:
            # Determine a default device if not specified
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self._metric = Metrics()
        # Temperature Scaling is applied to logits (or in case of XGBoost, probabilities converted to logits)
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature).to(self._device)

    def calibrate(self, cal_dataloader, fsc_group_index=None, alpha=None):
        """
        Calibrates the predictor using the calibration dataset with an XGBoost model.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
                                                           Expected to contain tensors that can be
                                                           converted to numpy arrays for the XGBoost model.
            alpha (float): The significance level. If None, uses self.alpha.
        """
        if alpha is None:
            alpha = self.alpha

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        # XGBoost models don't have an eval() mode like PyTorch models
        # self._model.eval() # Remove this line

        logits_list = []
        labels_list = []

        # Iterate through the dataloader to get calibration data
        with torch.no_grad(): # Still useful for any potential tensor operations
            for examples in cal_dataloader:
                # Assuming examples[0] is features and examples[1] is labels
                tmp_x_tensor = examples[0]
                if fsc_group_index is None:
                    tmp_labels_tensor = examples[1]
                else:
                    tmp_labels_tensor = examples[0, :, fsc_group_index]

                # Convert tensors to numpy arrays for XGBoost prediction
                tmp_x_np = tmp_x_tensor.cpu().numpy()
                tmp_labels_np = tmp_labels_tensor.cpu().numpy()

                # Get prediction probabilities from XGBoost model
                # model.predict_proba() returns a numpy array
                tmp_proba_np = self._model.predict_proba(tmp_x_np)

                # Convert probabilities back to tensors and move to the correct device
                tmp_logits = torch.tensor(tmp_proba_np, dtype=torch.float32).to(self._device)
                tmp_labels = torch.tensor(tmp_labels_np, dtype=torch.long).to(self._device) # Keep labels as long tensor


                tmp_calibrated_logits = self._logits_transformation(tmp_logits).detach()

                logits_list.append(tmp_calibrated_logits)
                labels_list.append(tmp_labels)

            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)

        # The rest of the calibration logic can remain similar, operating on tensors
        self.calculate_threshold(logits, labels, alpha)

    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances using the XGBoost model.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        # Convert the input tensor to a numpy array for XGBoost prediction
        x_batch_np = x_batch.cpu().numpy()

        # Get prediction probabilities from XGBoost model
        # model.predict_proba() returns a numpy array
        proba_np = self._model.predict_proba(x_batch_np)

        # Convert probabilities back to tensors and move to the correct device
        logits = torch.tensor(proba_np, dtype=torch.float32).to(self._device)

        # Apply temperature scaling (if temperature != 1) to the probabilities
        calibrated_logits = self._logits_transformation(logits).detach()

        # Generate prediction sets using the calibrated logits
        sets = self.predict_with_logits(calibrated_logits)
        return sets
