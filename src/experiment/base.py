from abc import abstractmethod
from typing import Literal

from equinox import Module

from src.sampling import AbstractSamplingAlgorithm
from src.voltammetry import VoltammetryType


class AbstractSamplingExperiment:
    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        noise: float,
        sigma: int,
        voltammetry_type: VoltammetryType,
        seed: int,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def true_parameters(self) -> Module:
        raise NotImplementedError
