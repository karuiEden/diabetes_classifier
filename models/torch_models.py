from pathlib import Path

import numpy as np
import torch


class TorchModel:
    def __init__(self, mode='cpu', tol=1e-5, l2=0, learning_rate=0.1, random_state=42, optim='adam'):
        self.optim = optim.lower()
        self.tol = tol
        self.l2 = l2
        self.lr = learning_rate
        use_gpu = torch.cuda.is_available() and mode == 'gpu'
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.model = None
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _build_model(self, n_features: int):
        self.model = torch.nn.Linear(n_features, 1).to(self.device)

    def _build_optimizer(self):
        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optim == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optim} not supported")

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        y = y.reshape(-1, 1)
        self._build_model(X.shape[1])
        self._build_optimizer()
        self.model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        err = self.tol + 1
        losses = [criterion(self.model(X), y).item()]
        while err > self.tol:
            if self.optim == 'lbfgs':
                def closure():
                    self.optimizer.zero_grad()
                    loss = criterion(self.model(X), y)
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
                losses.append(loss)
            else:
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = criterion(logits, y)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            err = abs(losses[-1] - losses[-2])
        return losses

    @torch.no_grad()
    def predict_prob(self, X):
        X = torch.tensor(X.to_numpy(), dtype=torch.float32, device=self.device)
        logits = self.model(X)
        return torch.sigmoid(logits).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) > threshold).astype(int)

    def get_weights(self):
        return self.model.weight.detach().cpu().numpy()

    @staticmethod
    def from_file(params_path: str, weights_path: str = None):
        p = Path(params_path)
        if not p.exists():
            raise FileNotFoundError(f"Hyper parameters file not found at {p}")
        model = torch.load(p, weights_only=False)
        return model

    def to_file(self, params_path: str, weights_path: str = None):
        p = Path(params_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, p)