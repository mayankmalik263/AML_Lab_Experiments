from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg


def numpy_demo() -> None:
    arr_1d = np.arange(1, 13)
    arr_2d = arr_1d.reshape(3, 4)
    arr_3d = arr_1d.reshape(2, 2, 3)

    print("NumPy 1D array:")
    print(arr_1d)
    print("\nNumPy 2D array (3, 4):")
    print(arr_2d)
    print("\nNumPy 3D array (2, 2, 3):")
    print(arr_3d)


def pandas_demo() -> None:
    series_a = pd.Series([10, 20, 30], index=["a", "b", "c"])
    series_b = pd.Series([2, 4, 5], index=["a", "b", "c"])
    division_result = series_a / series_b

    print("\nPandas Series A:")
    print(series_a)
    print("\nPandas Series B:")
    print(series_b)
    print("\nSeries division (A / B):")
    print(division_result)


def matplotlib_demo() -> None:
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 1, 8, 7]
    output_path = Path(__file__).with_name("exp_1_red_line_plot.png")

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color="red", marker="o")
    plt.title("Red Line Graph")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"\nMatplotlib plot saved as {output_path.name}")


def scipy_demo() -> None:
    matrix = np.array([[1, 2], [3, 4]])
    determinant = linalg.det(matrix)
    eigenvalues, eigenvectors = linalg.eig(matrix)

    print("\nSciPy matrix:")
    print(matrix)
    print(f"\nDeterminant: {determinant}")
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)


def statistics_demo() -> None:
    data = [2, 4, 4, 4, 5, 5, 7, 9]

    print("\nStatistics data:")
    print(data)
    print(f"Mean: {statistics.mean(data)}")
    print(f"Median: {statistics.median(data)}")
    print(f"Mode: {statistics.mode(data)}")
    print(f"Variance: {statistics.variance(data)}")
    print(f"Standard Deviation: {statistics.stdev(data)}")


def main() -> None:
    numpy_demo()
    pandas_demo()
    matplotlib_demo()
    scipy_demo()
    statistics_demo()


if __name__ == "__main__":
    main()
