import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from chronos import ChronosPipeline
from chronos.chronos import left_pad_and_stack_1D
from typing import Union, List

from aeon.classification.deep_learning import LITETimeClassifier
from aeon.classification.shapelet_based import SASTClassifier
from aeon.datasets import load_classification
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from time import time
import tqdm

import datasets, transformers
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader


class ChronosTransform(ChronosPipeline):
  def transform(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Transform the given time series using Chronos encoder.

        Parameters
        ----------
        context
            Input series. This is a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch.
        batch_size
            The size of the minibatches

        Returns
        -------
        embedding
            Tensor of embedding, of shape
            (batch_size, emb_size).
        """

        loader = DataLoader(context, batch_size = batch_size, shuffle=False)

        out = torch.cat([self.embed(x)[0][:,-1,:].view(x.shape[0], -1) for x in loader])

        return out

  def fit(self, X, y):
    """Nothing to fit since we are using a pretrained model"""
    return self


def get_dataset_from_tensors(X_train, y_train, X_test, y_test):
    unique_labels = np.unique(y_train)
    # map the labels from 0 and onwards
    label_map = {label: i for i, label in enumerate(unique_labels)}
    out_dict = {"features": [], "labels": []}
    eval_dict = {"features": [], "labels": []}
    for i in range(len(X_train)):
        out_dict["features"].append(X_train[i,:])
        out_dict["labels"].append(label_map[y_train[i]])
    train_dataset = datasets.Dataset.from_dict(out_dict)

    for i in range(len(X_test)):
        eval_dict["features"].append(X_test[i,:])
        eval_dict["labels"].append(label_map[y_test[i]])
    eval_dataset = datasets.Dataset.from_dict(eval_dict)

    # now put the datasets together
    dataset = datasets.DatasetDict({"train": train_dataset, "validation": eval_dataset})
    return dataset


if __name__ == "__main__":
    ds_name = "Chinatown"
    X_train, y_train = load_classification(ds_name, split="train")
    X_test, y_test = load_classification(ds_name, split="test")
    X_test = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    dataset = get_dataset_from_tensors(X_train, y_train, X_test, y_test)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    chronos_transform = ChronosTransform.from_pretrained(
        "amazon/chronos-t5-base",
        device_map=device,
        torch_dtype=torch.float16,
    )

    res = []
    batch_size = 16
    dataset_names = ["FordA", "FordB", 'Coffee', 'Chinatown', 'Fungi', 'FaceAll', 'InsectWingbeatSound', 'ElectricDevices']

    for ds_name in tqdm.tqdm(dataset_names, desc="Datasets"):
        print(f"Running on {ds_name}")
        X_train, y_train = load_classification(ds_name, split="train")
        X_test, y_test = load_classification(ds_name, split="test")

        rf_clf = RidgeClassifierCV()

        Xtrain_reshaped = X_train.reshape(X_train.shape[0], -1)
        Xtest_reshaped = X_test.reshape(X_test.shape[0], -1)

        # training
        start_fit = time()
        Xtrain_transf = chronos_transform.transform(Xtrain_reshaped, batch_size)
        rf_clf.fit(Xtrain_transf, y_train)

        # evaluating
        end_fit = time()
        Xtest_trainsf = chronos_transform.transform(Xtest_reshaped, batch_size)
        acc = rf_clf.score(Xtest_trainsf, y_test)
        end_score = time()

        res.append(['chronos', ds_name, acc, end_fit-start_fit, end_score - end_fit])


        res_df = pd.DataFrame(res, columns=["Model", "Dataset", "Accuracy", "TrainTime", "TestTime"])
