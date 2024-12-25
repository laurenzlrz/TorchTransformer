from abc import ABC, abstractmethod

import torch

class Masking(ABC):

    def __init__(self, replace_token):
        self.replace_token = replace_token
        pass

    @abstractmethod
    def mask(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BertMasking(Masking):

    def __init__(self, replace_token, probability: float):
        super().__init__(replace_token)
        self.probability = probability


    def mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.size()) < self.probability
        x[mask] = self.replace_token
        return x