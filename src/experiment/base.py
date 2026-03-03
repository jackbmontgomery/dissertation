from abc import abstractmethod

from equinox import Module

from src.sampling import AbstractSamplingAlgorithm


class AbstractSamplingExperiment:
    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        noise: float,
        sigma: int,
        seed: int,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def true_parameters(self) -> Module:
        raise NotImplementedError
