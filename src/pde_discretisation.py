# from abc import abstractmethod
#
# from equinox import Module
# from jaxtyping import Array
#
#
# class AbstractFDMDiscretization(Module):
#     @abstractmethod
#     def A(self) -> Array:
#         raise NotImplementedError
#
#     @abstractmethod
#     def b(self) -> Array:
#         raise NotImplementedError
#
#
# class ButlerVolmerFDMDiscretization1D(Module):
#     def A(self, X: Array, theta: float, params: ButlerVolmerParameters) -> Array:
#         raise NotImplementedError
#
#     def b(self) -> Array:
#         raise NotImplementedError
