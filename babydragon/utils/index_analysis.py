import numpy as np
from typing import Any, Dict

class EmbeddingAnalysis:
    def __init__(self, kernel_matrix: np.ndarray) -> None:
        """
        Initializes an instance of the EmbeddingAnalysis class.

        Parameters:
        kernel_matrix (np.ndarray): A square kernel matrix.

        Raises:
        ValueError: If the input kernel matrix is not square or symmetric.
        """
        if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
            raise ValueError("The input kernel matrix must be square.")
        if not np.allclose(kernel_matrix, kernel_matrix.T, atol=1e-8):
            raise ValueError("The input kernel matrix must be symmetric.")
        self.kernel_matrix = kernel_matrix
        self.eigenvalues = np.linalg.eigvalsh(kernel_matrix)

    def check_symmetry(self) -> bool:
        """
        Checks if the kernel matrix is symmetric.

        Returns:
        bool: True if the kernel matrix is symmetric, False otherwise.
        """
        return np.allclose(self.kernel_matrix, self.kernel_matrix.T, atol=1e-8)

    def is_positive_definite(self) -> bool:
        """
        Checks if the kernel matrix is positive definite.

        Returns:
        bool: True if all eigenvalues are positive, False otherwise.
        """
        return np.all(self.eigenvalues > 0)

    def is_positive_semi_definite(self) -> bool:
        """
        Checks if the kernel matrix is positive semi-definite.

        Returns:
        bool: True if all eigenvalues are non-negative, False otherwise.
        """
        return np.all(self.eigenvalues >= 0)

    def is_negative_definite(self) -> bool:
        """
        Checks if the kernel matrix is negative definite.

        Returns:
        bool: True if all eigenvalues are negative, False otherwise.
        """
        return np.all(self.eigenvalues < 0)

    def is_negative_semi_definite(self) -> bool:
        """
        Checks if the kernel matrix is negative semi-definite.

        Returns:
        bool: True if all eigenvalues are non-positive, False otherwise.
        """
        return np.all(self.eigenvalues <= 0)

    def is_indefinite(self) -> bool:
        """
        Checks if the kernel matrix is indefinite.

        Returns:
        bool: True if the matrix has both positive and negative eigenvalues, False otherwise.
        """
        has_negative = np.any(self.eigenvalues < 0)
        has_non_negative = np.any(self.eigenvalues >= 0)
        return has_negative and has_non_negative

    def check_definiteness(self, num_random_vectors: int = 1000) -> Dict[str, bool]:
        """
        Checks the definiteness of the kernel matrix using random vectors.

        num_random_vectors: Number of random vectors to use for checking.

        Returns:
        Dict[str, bool]: A dictionary with the results of the analysis.
        """
        n = self.kernel_matrix.shape[0]
        is_positive_definite = True
        is_negative_definite = True
        for _ in range(num_random_vectors):
            x = np.random.randn(n)
            xTMx = x.T @ self.kernel_matrix @ x
            if xTMx <= 0:
                is_positive_definite = False
            if xTMx >= 0:
                is_negative_definite = False
        return {
            "is_positive_definite": is_positive_definite,
            "is_negative_definite": is_negative_definite,
        }

    def run_analysis(self) -> Dict[str, bool]:
        return {
            "is_symmetric": self.check_symmetry(),
            "is_positive_semi_definite": self.is_positive_semi_definite(),
            "is_negative_semi_definite": self.is_negative_semi_definite(),
            "is_indefinite": self.is_indefinite(),
            **self.check_definiteness()
        }
