import torch
import syft as sy
import numpy as np
import math

class EnhancedReplicatedSecretSharing:
    def __init__(self, n_parties, crypto_provider=None):
        self.n_parties = n_parties
        self.hook = sy.TorchHook(torch)  # initialize PySyft hook
        self.parties = [sy.VirtualWorker(self.hook, id=f"worker_{i}") for i in range(self.n_parties)]
        self.crypto_provider = crypto_provider if crypto_provider else sy.VirtualWorker(self.hook, id='crypto_provider')  # create a virtual worker for crypto functions

    def share_tensor(self, tensor):
        """
        Shares a tensor across multiple parties using replicated secret sharing
        :param tensor: PyTorch tensor to be shared
        :return: Shared tensor
        """
        tensor_ptrs = sy.MultiPointerTensor([tensor], shape=(tensor.shape[0], tensor.shape[1], self.n_parties)).share(*self.parties, crypto_provider=self.crypto_provider)
        return tensor_ptrs

    def reveal(self, tensor_ptrs):
        """
        Reveal the shared tensor from the secret shares.
        :param tensor_ptrs: The pointer to the shared tensor
        :return: The revealed tensor
        """
        revealed_tensor = tensor_ptrs.get().sum(dim=2)  # Aggregate all shares
        return revealed_tensor

    def add(self, tensor_ptrs1, tensor_ptrs2):
        """
        Add two shared tensors securely
        :param tensor_ptrs1: The first shared tensor
        :param tensor_ptrs2: The second shared tensor
        :return: The result of addition in the shared domain
        """
        result_ptrs = tensor_ptrs1 + tensor_ptrs2
        return result_ptrs

    def mul(self, tensor_ptrs1, tensor_ptrs2):
        """
        Multiply two shared tensors securely
        :param tensor_ptrs1: The first shared tensor
        :param tensor_ptrs2: The second shared tensor
        :return: The result of multiplication in the shared domain
        """
        result_ptrs = tensor_ptrs1 * tensor_ptrs2
        return result_ptrs

    def isomorphic(self, matrix_ptrs1, matrix_ptrs2):
        """
        Check whether two adjacency matrices are isomorphic using encrypted matrices.
        :param matrix_ptrs1: The first matrix pointer
        :param matrix_ptrs2: The second matrix pointer
        :return: Boolean result indicating isomorphism
        """
        matrix1 = matrix_ptrs1.get().sum(dim=2)
        matrix2 = matrix_ptrs2.get().sum(dim=2)
        G1 = nx.from_numpy_array(matrix1.cpu().numpy())
        G2 = nx.from_numpy_array(matrix2.cpu().numpy())
        result = nx.is_isomorphic(G1, G2)
        return result
