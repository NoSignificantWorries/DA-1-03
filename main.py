import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


def main(point_of_interest: str,
         show: bool = False,
         save: bool = True,
         save_path: str = "results.png",
         dpi: int = 300,
         figsize: tuple[int, int] = (24, 16)) -> None:
    """
    Plot histogram and boxplot for a selected feature from the diabetes dataset.

    Parameters:
        point_of_interest (str): Column name to visualize. Must be one of the dataset features or "target".
        show (bool, optional): If True, display the plot window. Defaults to False.
        save (bool, optional): If True, save the plot to save_path. Defaults to True.
        save_path (str, optional): File path to save the plot image. Defaults to "results.png".
        dpi (int, optional): Output image DPI (minimum 100). Defaults to 300.
        figsize (tuple[int, int], optional): Figure size in inches (width, height), minimum 2x2. Defaults to (24, 16).

    Raises:
        ValueError: If dpi < 100, figsize is smaller than 2x2, or point_of_interest is not a valid column.
    """
    
    # cheking integer values
    if dpi < 100:
        raise ValueError("dpi must be at least 100")
    if figsize[0] < 2 or figsize[1] < 2:
        raise ValueError("figsize must be at least 2x2")

    # loading dataset
    dataset = sklearn.datasets.load_diabetes()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target

    # checking if the point_of_interest is valid
    columns = df.columns
    if point_of_interest not in columns:
        raise ValueError(f"\"point_of_interest\" must be one of:\n- {"\n- ".join(columns)}")
    
    current_column = df[point_of_interest]
    
    # culculating IQR and median value
    median = np.median(current_column)
    q1 = np.percentile(current_column, 25)
    q3 = np.percentile(current_column, 75)
    iqr = q3 - q1

    # start plotting
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    fig.suptitle(f"Distribution of '{point_of_interest}' (n={len(current_column)}) - Diabetes dataset", fontsize=18, fontweight="bold")

    # creating hist plot of data
    ax[0].set_title(f"Histogram of '{point_of_interest}'")
    ax[0].hist(current_column, bins=30, color="green", edgecolor="black", alpha=0.7, label=f"{point_of_interest} (IQR: {iqr:.3f})")
    # adding vertical lines for median, q1, and q3
    ax[0].axvline(x=median, color="red", linestyle="--", linewidth=4, label=f"median: {median:.3f}")
    ax[0].axvline(x=q1, color="black", linestyle="--", linewidth=2, label=f"Q1: {q1:.3f}")
    ax[0].axvline(x=q3, color="blue", linestyle="--", linewidth=2, label=f"Q3: {q3:.3f}")
    # displaying legend
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.3)

    # creating boxplot
    ax[1].set_title(f"Boxplot of '{point_of_interest}'")
    ax[1].boxplot(current_column, vert=False, notch=True)
    ax[1].set_xlabel("Value")
    ax[1].set_yticks([1])
    ax[1].set_yticklabels([f"{point_of_interest}"])
    ax[1].grid(True, axis="x", linestyle="--", alpha=0.3)

    # adjust layout to accommodate the suptitle
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # save plot if needed
    if save:
        fig.savefig(save_path, dpi=dpi)
    
    # showing plot if needed
    if show:
        plt.show()


if __name__ == "__main__":
    # CLI argument parsing and starting main func
    # Prepare CLI parser
    parser = argparse.ArgumentParser(
        description="Visualize distribution (histogram and boxplot) for a selected feature from the diabetes dataset."
    )

    # Available columns: dataset features plus 'target'
    dataset = sklearn.datasets.load_diabetes()
    available_columns = list(dataset.feature_names) + ["target"]

    parser.add_argument(
        "-p", "--poi",
        dest="point_of_interest",
        choices=available_columns,
        required=True,
        help="Point of interest (column) to visualize. Choices: %(choices)s"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show plot window."
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        default=True,
        help="Do not save the plot to a file."
    )
    parser.add_argument(
        "--save-path",
        default="results.png",
        help="File path to save the plot image."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI (min 100)."
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(24, 16),
        help="Figure size in inches as two integers: WIDTH HEIGHT. Default: 24 16."
    )

    args = parser.parse_args()

    try:

        main(
            point_of_interest=args.point_of_interest,
            show=args.show,
            save=args.save,
            save_path=args.save_path,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
        )
    except BaseException as val_err:
        print(f"ERROR while running main function:\n{val_err}")
