from abc import abstractmethod

from equinox import Module

from src.sampling import AbstractSamplingAlgorithm
from src.voltammetry import AbstractVoltammetryTechnique


class AbstractSamplingExperiment:
    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        voltammetry: AbstractVoltammetryTechnique,
        seed: int,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def true_parameters(self) -> Module:
        raise NotImplementedError
