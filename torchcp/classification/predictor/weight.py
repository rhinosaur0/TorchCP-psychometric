# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchcp.classification.predictor.split import SplitPredictor
from torchcp.classification.predictor.utils import build_DomainDetecor, IW
from torchcp.classification.utils import ConfCalibrator
from torchcp.classification.utils.metrics import Metrics


class WeightedPredictor(SplitPredictor):
    """
    Method: Weighted Conformal Prediction
    Paper: Conformal Prediction Under Covariate Shift (Tibshirani et al., 2019)
    Link: https://arxiv.org/abs/1904.06019
    Github: https://github.com/ryantibs/conformal/
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module): A PyTorch model.
        alpha (float, optional): The significance level. Default is 0.1.
        image_encoder (torch.nn.Module): A PyTorch model to generate the embedding feature of an input image.
        domain_classifier (torch.nn.Module, optional): A PyTorch model (a binary classifier) to predict the probability that an embedding feature comes from the source domain. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        device (torch.device, optional): The device on which the model is located. Default is None.
    """

    def __init__(self, score_function, model=None, temperature=1, alpha=0.1, image_encoder=None, domain_classifier=None, device=None):

        super().__init__(score_function, model, temperature, alpha, device)

        if image_encoder is None:
            raise ValueError("image_encoder cannot be None.")

        self.image_encoder = image_encoder.to(self._device)
        self.domain_classifier = domain_classifier

        #  non-conformity scores
        self.scores = None
        # significance level
        self.alpha = None
        # Domain Classifier

    def calibrate(self, cal_dataloader, alpha=None):
        """
        Calibrate the model using the calibration set.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
            alpha (float): The significance level. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        logits_list = []
        labels_list = []
        cal_features_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                cal_features_list.append(self.image_encoder(tmp_x))
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
            self.source_image_features = torch.cat(cal_features_list).float()

        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha=None):
        """
        Calculate the conformal prediction threshold.

        Args:
            logits (torch.Tensor): The logits output from the model.
            labels (torch.Tensor): The ground truth labels.
            alpha (float): The significance level. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        self.alpha = alpha
        self.scores = torch.zeros(logits.shape[0] + 1).to(self._device)
        self.scores[:logits.shape[0]] = self.score_function(logits, labels)
        self.scores[logits.shape[0]] = torch.tensor(torch.inf).to(self._device)
        self.scores_sorted = self.scores.sort()[0]

    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """
        if not hasattr(self, "scores_sorted"):
            raise ValueError("Please calibrate first to get self.scores_sorted.")

        bs = x_batch.shape[0]
        with torch.no_grad():
            image_features = self.image_encoder(x_batch.to(self._device)).float()
            w_new = self.IW(image_features)

        w_sorted = self.w_sorted.expand([bs, -1])
        w_sorted = torch.cat([w_sorted, w_new.unsqueeze(1)], 1)
        p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
        p_sorted_acc = p_sorted.cumsum(1)

        i_T = torch.argmax((p_sorted_acc >= 1.0 - self.alpha).int(), dim=1, keepdim=True)
        q_hat_batch = self.scores_sorted.expand([bs, -1]).gather(1, i_T).detach()

        logits = self._model(x_batch.to(self._device)).float()
        logits = self._logits_transformation(logits).detach()
        predictions_sets_list = []
        for index, (logits_instance, q_hat) in enumerate(zip(logits, q_hat_batch)):
            predictions_sets_list.append(self.predict_with_logits(logits_instance, q_hat))

        predictions_sets = torch.cat(predictions_sets_list, dim=0)  # (N_val x C)
        return predictions_sets

    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset using domain adaptation.

        This method trains a domain classifier if not provided, computes importance 
        weights for validation set, generates prediction sets and calculates metrics.

        Args:
            val_dataloader (DataLoader): Dataloader for validation set.

        Returns:
            dict: Dictionary containing evaluation metrics:
                - Coverage_rate: Empirical coverage rate on validation set
                - Average_size: Average size of prediction sets

        Raises:
            ValueError: If calibration has not been performed first.
        """
        # Extract features from validation set
        self._model.eval()
        features_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch[0].to(self._device)
                features = self.image_encoder(inputs)
                features_list.append(features)
        target_features = torch.cat(features_list, dim=0).float()  # (N_val x D)

        # Train domain classifier if needed
        if not hasattr(self, "source_image_features"):
            raise ValueError("Please calibrate first to get source_image_features.")

        if self.domain_classifier is None:
            self._train_domain_classifier(target_features)

        # Compute importance weights
        self.IW = IW(self.domain_classifier).to(self._device)
        weights_cal = self.IW(self.source_image_features.to(self._device))
        self.w_sorted = torch.sort(weights_cal, descending=False)[0]

        # Generate predictions
        predictions_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device and get predictions
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device)

                # Get predictions as bool tensor (N x C)
                batch_predictions = self.predict(inputs)

                # Accumulate predictions and labels
                predictions_list.append(batch_predictions)
                labels_list.append(labels)

        # Concatenate all batches
        val_predictions = torch.cat(predictions_list, dim=0)  # (N_val x C) 
        val_labels = torch.cat(labels_list, dim=0)  # (N_val,)

        # Compute evaluation metrics
        metrics = {
            "coverage_rate": self._metric('coverage_rate')(val_predictions, val_labels),
            "average_size": self._metric('average_size')(val_predictions, val_labels)
        }

        return metrics

    def _train_domain_classifier(self, target_image_features):
        source_labels = torch.zeros(self.source_image_features.shape[0]).to(self._device)
        target_labels = torch.ones(target_image_features.shape[0]).to(self._device)

        input = torch.cat((self.source_image_features, target_image_features))
        labels = torch.cat((source_labels, target_labels))
        dataset = torch.utils.data.TensorDataset(input.float(), labels.float().long())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=False)

        self.domain_classifier = build_DomainDetecor(target_image_features.shape[1], 2, self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.001)

        epochs = 5
        for epoch in range(epochs):
            loss_log = 0
            accuracy_log = 0
            for X_train, y_train in data_loader:
                y_train = y_train.to(self._device)
                outputs = self.domain_classifier(X_train.to(self._device))
                loss = criterion(outputs, y_train.view(-1))
                loss_log += loss.item() / len(data_loader)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum((predictions == y_train.view(-1))).item() / len(y_train)
                accuracy_log += accuracy / len(data_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class XGBoostWeightedPredictor(SplitPredictor):
    """
    XGBoost version of Weighted Conformal Prediction
    Method: Weighted Conformal Prediction
    Paper: Conformal Prediction Under Covariate Shift (Tibshirani et al., 2019)
    
    Args:
        score_function (callable): Non-conformity score function.
        model: XGBoost model.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        alpha (float, optional): The significance level. Default is 0.1.
        device (torch.device, optional): The device for tensor operations. Default is None.
        domain_classifier: Pre-trained domain classifier for computing importance weights.
    """

    def __init__(self, score_function, model, temperature=1, alpha=0.1, device=None, domain_classifier=None):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        self.score_function = score_function
        self._model = model  # XGBoost model

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")
        self.alpha = alpha

        # For XGBoost, device is less relevant for the model itself,
        # but we keep it for tensor operations later.
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._metric = Metrics()
        # Temperature Scaling applied to probabilities converted to logits
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature).to(self._device)
        
        # Domain classifier for importance weighting
        if domain_classifier is None:
            raise ValueError("domain_classifier cannot be None for weighted prediction.")
        self.domain_classifier = domain_classifier
        self.IW = IW(self.domain_classifier).to(self._device)
        
        # Store calibration features and weights
        self.source_features = None
        self.w_sorted = None
        self.scores_sorted = None

    def calibrate(self, cal_dataloader, group_index, alpha=None):
        """
        Calibrate the model using the calibration set with importance weighting.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
            group_index (int): The index of the group label feature in the input.
            alpha (float): The significance level. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self.alpha = alpha
        logits_list = []
        labels_list = []
        group_labels_list = []

        # Iterate through calibration dataloader
        with torch.no_grad():
            for cal_batch in cal_dataloader:
                # Get features and labels from calibration set
                tmp_x_tensor = cal_batch[0]
                tmp_labels_tensor = cal_batch[1]

                # Extract group labels
                group_labels = tmp_x_tensor[:, group_index]
                group_labels_list.append(group_labels)

                # Convert tensors to numpy arrays for XGBoost prediction
                tmp_x_np = tmp_x_tensor.cpu().numpy()
                tmp_labels_np = tmp_labels_tensor.cpu().numpy()

                # Get prediction probabilities from XGBoost model
                tmp_proba_np = self._model.predict_proba(tmp_x_np)

                # Convert probabilities back to tensors and move to the correct device
                tmp_logits = torch.tensor(tmp_proba_np, dtype=torch.float32).to(self._device)
                tmp_labels = torch.tensor(tmp_labels_np, dtype=torch.long).to(self._device)

                # Apply temperature scaling
                tmp_calibrated_logits = self._logits_transformation(tmp_logits).detach()

                logits_list.append(tmp_calibrated_logits)
                labels_list.append(tmp_labels)

            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
            self.cal_group_labels = torch.cat(group_labels_list).cpu().numpy()

        # Compute non-conformity scores for calibration set
        self.cal_scores = self.score_function(logits, labels).cpu().numpy()
        
        # Store calibration set size for computing group densities
        self.cal_size = len(self.cal_scores)

    def evaluate(self, val_dataloader: DataLoader, group_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset using group-based weighted conformal prediction.

        This method computes group-specific importance weights for validation set, generates prediction sets 
        using weighted quantiles, and calculates coverage and size metrics.

        Args:
            val_dataloader (DataLoader): Dataloader for validation set.
            group_index (int): Index of the group label in the input features.
            
        Returns:
            dict: Dictionary containing evaluation metrics:
                - coverage_rate: Empirical coverage rate on validation set
                - average_size: Average size of prediction sets
        """
        import numpy as np
        
        if not hasattr(self, 'cal_scores'):
            raise ValueError("Please calibrate first to get calibration scores.")
        
        # Collect validation data
        val_features_list = []
        val_labels_list = []
        val_group_labels_list = []

        for batch, groups in zip(val_dataloader, group_dataloader):
            val_features_list.append(batch[0])
            val_labels_list.append(batch[1])
            val_group_labels_list.append(groups)

        val_features = torch.cat(val_features_list, dim=0)
        val_labels = torch.cat(val_labels_list, dim=0)
        val_group_labels = torch.cat(val_group_labels_list, dim=0).cpu().numpy()
        
        val_size = len(val_labels)
        
        # Compute group counts for calibration and validation sets
        unique_groups = np.unique(np.concatenate([self.cal_group_labels, val_group_labels]))
        cal_group_counts = {g: np.sum(self.cal_group_labels == g) for g in unique_groups}
        val_group_counts = {g: np.sum(val_group_labels == g) for g in unique_groups}
        
        # Compute calibration weights: w_cal[i] = p_test(group_i) / p_cal(group_i)
        w_calibration = np.zeros(len(self.cal_scores))
        for i, group in enumerate(self.cal_group_labels):
            p_cal = cal_group_counts[group] / self.cal_size
            p_test = val_group_counts.get(group, 0) / val_size
            w_calibration[i] = p_test / p_cal if p_cal > 0 else 0
        
        # Get predictions from model
        val_features_np = val_features.cpu().numpy()
        val_proba_np = self._model.predict_proba(val_features_np)
        val_logits = torch.tensor(val_proba_np, dtype=torch.float32).to(self._device)
        val_logits = self._logits_transformation(val_logits).detach()
        
        # Compute non-conformity scores for validation set
        val_scores = self.score_function(val_logits, val_labels.to(self._device)).cpu().numpy()

        print("Validation scores:", val_scores)
        # Compute weighted quantiles for each validation sample based on its group
        predictions_list = []
        for i in range(val_size):
            group = val_group_labels[i]
            
            # Compute w_test for this sample's group
            p_cal = cal_group_counts[group] / self.cal_size
            p_test = val_group_counts[group] / val_size
            w_test = p_test / p_cal if p_cal > 0 else 1.0
            
            # Compute weighted quantile using the formula from the example
            s = np.sum(w_calibration) + w_test
            new_p = w_calibration / s
            
            # Sort calibration scores and corresponding weights
            indices = np.argsort(self.cal_scores)
            cal_scores_sorted = self.cal_scores[indices]
            new_p_sorted = new_p[indices]
            
            # Find the weighted quantile
            cumsum = np.cumsum(new_p_sorted)
            quantile_idx = np.searchsorted(cumsum, 1.0 - self.alpha)
            quantile_idx = min(quantile_idx, len(cal_scores_sorted) - 1)
            q_value = cal_scores_sorted[quantile_idx]
            
            # Generate prediction set: include all classes where score <= quantile
            q_value_tensor = torch.tensor(q_value, dtype=torch.float32).to(self._device)
            pred_set = self.predict_with_logits(val_logits[i], q_value_tensor)
            predictions_list.append(pred_set)
        
        val_predictions = torch.cat(predictions_list, dim=0)
        
        # Compute evaluation metrics
        metrics = {
            "coverage_rate": self._metric('coverage_rate')(val_predictions, val_labels.to(self._device)),
            "average_size": self._metric('average_size')(val_predictions, val_labels.to(self._device))
        }
        
        return metrics


