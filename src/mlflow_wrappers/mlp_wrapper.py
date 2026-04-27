# src/mlflow_wrappers/mlp_wrapper.py

import mlflow.pyfunc


class MLPWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import torch
        from models.landmark_mlp import build_mlp

        checkpoint = torch.load(context.artifacts["model_path"], map_location="cpu")

        self.model = build_mlp(checkpoint["cfg"], len(checkpoint["classes"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.classes = checkpoint["classes"]

    def predict(self, context, model_input):
        import torch
        import numpy as np
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            x = model_input.values.astype(np.float32)
        else:
            x = np.array(model_input).astype(np.float32)

        x = torch.tensor(x)

        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).numpy()

        return [self.classes[i] for i in preds]