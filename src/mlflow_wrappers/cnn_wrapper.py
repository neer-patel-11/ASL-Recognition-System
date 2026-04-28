# src/mlflow_wrappers/cnn_wrapper.py

import mlflow.pyfunc


class CNNWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import torch
        from models.tiny_cnn import build_cnn

        checkpoint = torch.load(context.artifacts["model_path"], map_location="cpu")

        self.model = build_cnn(
            checkpoint["cfg"],
            len(checkpoint["classes"])
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.classes = checkpoint["classes"]

    def predict(self, context, model_input):
        import torch
        import numpy as np

        x = np.array(model_input, dtype=np.float32)
        x = torch.tensor(x)

        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).numpy()
            print("We are here in the wrapper")
            print("Classes:", self.classes)
            print("Logits:", logits)
            print("Preds:", preds)
        return [self.classes[i] for i in preds]