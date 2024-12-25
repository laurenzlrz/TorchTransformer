from abc import ABC, abstractmethod

import torch

class Masking(ABC):

    def __init__(self, replace_token):
        self.replace_token = replace_token
        self.training = False
        pass

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.applyMask(x)
        return x

    @abstractmethod
    def applyMask(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BertMasking(Masking):

    def __init__(self, replace_token, probability: float):
        super().__init__(replace_token)
        self.probability = probability


    def applyMask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.size()) < self.probability
        x[mask] = self.replace_token
        return x