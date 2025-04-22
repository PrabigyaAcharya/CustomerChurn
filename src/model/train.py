import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

from src.model.network import ChurnNet

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import tempfile

import json

from src.pipeline import preprocess, feature_engineering

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") 

with open('src/config.json', 'r') as f:
    config = json.load(f)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() #
    acc = (correct / len(y_pred)) * 100 
    return acc

def train_from_csv():
    df = pd.read_csv('../data/processed.csv')
    df_new = preprocess(df)
    X_train, y_train, X_test, y_test = feature_engineering(df_new)
    return train_model(X_train, y_train, X_test, y_test)

def train_model(X_train, y_train, X_test, y_test):

    model = ChurnNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = config["model"]["lr"])


    with mlflow.start_run():
        mlflow.log_params({
            "lr" : config["model"]["lr"],
            "batch_size": config["model"]["batch_size"],
            "epochs" : config["model"]["epochs"],
        })



        for epoch in range(config["model"]["epochs"]):
            model.train()
            y_logits = model(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
            y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
            loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
            acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
            optimizer.zero_grad()

    # 4. Loss backwards
            loss.backward()

    # 5. Optimizer step
            optimizer.step()

            model.eval()

            input_example = X_train[:1]
            output_example = model(input_example)
            input_example_np = input_example.detach().numpy()
            output_example_np = output_example.detach().numpy()


            # Infer model signature
            signature = infer_signature(
                input_example_np, 
                output_example_np
            )

        # Log model with input example and signature
            mlflow.pytorch.log_model(
                model, 
                artifact_path="model",
                input_example=input_example_np,
                signature=signature
            )

            with torch.inference_mode():
        # 1. Forward pass
                test_logits = model(X_test).squeeze() 
                test_pred = torch.round(torch.sigmoid(test_logits))
                test_probs = torch.sigmoid(test_logits).detach().numpy()
                test_pred_np = test_pred.numpy()
        # 2. Caculate loss/accuracy
                test_loss = loss_fn(test_logits,
                            y_test)
                test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
                
            # Generate predictions for the test set
                y_true = y_test.detach().numpy()
                y_pred_np = test_pred.detach().numpy()

            precision = precision_score(y_true, test_pred_np)
            recall    = recall_score(y_true, test_pred_np)
            f1        = f1_score(y_true, test_pred_np)
            roc_auc   = roc_auc_score(y_true, test_probs)
            # Compute confusion matrix

            mlflow.log_metrics({
                "train_loss": loss.item(),
                "train_accuracy": acc,
                "test_loss": loss_fn(test_logits, y_test).item(),
                "test_accuracy": accuracy_fn(y_test, test_pred),
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_roc_auc": roc_auc,
            }, step=epoch)


            cm = confusion_matrix(y_true, y_pred_np)

            # Plot and log the confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()

            # Save it to a temp file and log to MLflow
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name)
                mlflow.log_artifact(tmp.name, artifact_path="plots")
                plt.close()


    # Print out what's happening every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        
        mlflow.pytorch.log_model(model, "model", input_example=input_example_np, signature=signature)