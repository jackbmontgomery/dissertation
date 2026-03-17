from abc import ABC, abstractmethod

from chex import PRNGKey

from src.params import Params


class AbstractReaction(ABC):
    @property
    @abstractmethod
    def true_parameters(self) -> Params:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_init_params(self, key: PRNGKey, num: int) -> Params:
        raise NotImplementedError
