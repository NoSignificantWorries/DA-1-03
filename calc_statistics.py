import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


class PlotCore:
    """
    Lightweight wrapper over Matplotlib to standardize multi-axes plotting.
    Provides safe axis selection and common plot helpers.
    """

    def __init__(self,
                 nrows: int = 1,
                 ncols: int = 1,
                 figsize: tuple[int, int] = (24, 16),
                 title: str = "") -> None:

        # validating figure size values (must be at least 2x2)
        if figsize[0] < 2 or figsize[1] < 2:
            raise ValueError("figsize must be at least 2x2")
        # validating grid shape
        if nrows < 1 or ncols < 1:
            raise ValueError("nrows and ncols must be >= 1")

        self.figsize = figsize
        self.nrows = nrows
        self.ncols = ncols

        self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)

        # add a common title if provided
        if bool(title):
            self.fig.suptitle(title)

    def _test_position(self, irow: int, icol: int) -> None:
        """
        Validate axis indices against the configured grid.
        """
        if irow < 0 or icol < 0:
            raise ValueError("ERROR: Invalid plot position indices.")

        if irow + 1 > self.nrows or icol + 1 > self.ncols:
            raise ValueError("ERROR: Too big plot position indices.")

    def _get_axis(self, irow: int, icol: int):
        """
        Resolve the Matplotlib axis object given row/col (0-based indices).
        """
        self._test_position(irow, icol)

        # handle all possible shapes returned by plt.subplots
        if self.nrows == 1 and self.ncols == 1:
            return self.axes
        if self.ncols == 1 and self.nrows > 1:
            return self.axes[irow]
        if self.nrows == 1 and self.ncols > 1:
            return self.axes[icol]
        return self.axes[irow][icol]

    def _ensure_position_and_draw(self, irow: int, icol: int, func: str, **kwargs) -> None:
        """
        Get an axis by position and call a specified Matplotlib Axes method.
        """
        ax = self._get_axis(irow, icol)
        method = getattr(ax, func)
        method(**kwargs)

    def add_hist(self,
                 data_to_show: np.ndarray,
                 bins: int = 30,
                 color: str = "green",
                 edgecolor: str = "black",
                 alpha: float = 1.0,
                 label: str = "Hist",
                 irow: int = 0,
                 icol: int = 0) -> None:
        """
        Add a histogram to a specified subplot.
        """
        self._ensure_position_and_draw(
            irow, icol, "hist",
            x=data_to_show,
            bins=bins,
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            label=label,
        )

    def add_boxplot(self,
                    data_to_show: np.ndarray,
                    vert: bool = False,
                    notch: bool = True,
                    irow: int = 0,
                    icol: int = 0) -> None:
        """
        Add a boxplot to a specified subplot.
        """
        self._ensure_position_and_draw(irow, icol, "boxplot", x=data_to_show, vert=vert, notch=notch)

    def add_vline(self,
                  irow: int = 0,
                  icol: int = 0,
                  value: float = 0,
                  color: str = "black",
                  style: str = "--",
                  width: float = 2.0,
                  label: str = "axvline") -> None:
        """
        Add a vertical reference line to a specified subplot.
        """
        self._ensure_position_and_draw(
            irow, icol, "axvline",
            x=value, color=color, linestyle=style, linewidth=width, label=label
        )

    # convenient alias to tolerate plural naming in callers
    def add_vlines(self, irow: int = 0, icol: int = 0, value: float = 0, color: str = "black",
                   style: str = "--", width: float = 2.0, label: str = "axvline") -> None:
        self.add_vline(irow=irow, icol=icol, value=value, color=color, style=style, width=width, label=label)

    def add_title(self, irow: int = 0, icol: int = 0, title: str = "") -> None:
        """
        Set a subplot title.
        """
        self._ensure_position_and_draw(irow, icol, "set_title", label=title)

    def add_oxy_labels(self, irow: int = 0, icol: int = 0, xlabel: str = "", ylabel: str = "") -> None:
        """
        Set x/y axis labels if provided.
        """
        if bool(xlabel):
            self._ensure_position_and_draw(irow, icol, "set_xlabel", xlabel=xlabel)
        if bool(ylabel):
            self._ensure_position_and_draw(irow, icol, "set_ylabel", ylabel=ylabel)

    def add_grid(self, irow: int = 0, icol: int = 0, axis: str = "both", linestyle: str = "--", alpha: float = 0.3) -> None:
        """
        Enable subplot grid with styling.
        """
        self._ensure_position_and_draw(irow, icol, "grid", visible=True, axis=axis, linestyle=linestyle, alpha=alpha)

    def add_legend(self, irow: int = 0, icol: int = 0) -> None:
        """
        Show legend on the specified subplot.
        """
        self._ensure_position_and_draw(irow, icol, "legend")

    def save_plot(self, path: str = "results.png", dpi: int = 300) -> None:
        """
        Save figure to a file. DPI must be at least 100.
        """
        if dpi < 100:
            raise ValueError("dpi must be at least 100")
        # improve layout before saving (keep room for suptitle)
        try:
            self.fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        except Exception:
            # tight_layout may fail in some backends; ignore and continue
            pass
        self.fig.savefig(path, dpi=dpi)

    def show(self) -> None:
        """
        Display the figure window.
        """
        try:
            self.fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        except Exception:
            pass
        plt.show()


def load_dataset() -> pd.DataFrame:
    """
    Load the diabetes dataset from sklearn.
    """
    dataset = sklearn.datasets.load_diabetes()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target

    return df


def calculate_MIQR(data: np.ndarray) -> tuple[float, float, float, float]:
    """
    Calculate the Median, the Quantiles and the Interquartile Range (IQR) of a given numpy array.

    Parameters:
        data (np.ndarray): The numpy array to calculate the IQR and Median for.
    """

    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    return median, q1, q3, iqr


def calc_statistics(
        point_of_interest: str,
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

    df = load_dataset()

    # checking if the point_of_interest is valid
    columns = df.columns
    if point_of_interest not in columns:
        choices = "\n- ".join(columns)
        raise ValueError("\"point_of_interest\" must be one of:\n- " + choices)

    current_column = df[point_of_interest].values

    # compute summary statistics for reference lines and annotations
    median, q1, q3, iqr = calculate_MIQR(current_column)

    # core plotting: 2 rows (hist + boxplot), 1 column
    plotter = PlotCore(
        nrows=2,
        ncols=1,
        figsize=figsize,
        title=f"Distribution of '{point_of_interest}' (n={len(current_column)}) - Diabetes dataset",
    )

    # creating hist plot of data
    plotter.add_title(0, 0, f"Histogram of '{point_of_interest}'")
    plotter.add_hist(
        current_column,
        bins=30,
        color="green",
        edgecolor="black",
        alpha=0.7,
        label=f"{point_of_interest} (IQR: {iqr:.3f})",
        irow=0,
        icol=0,
    )

    # adding vertical lines for median, q1, and q3
    plotter.add_vline(0, 0, median, "red", "--", 4.0, f"median: {median:.3f}")
    plotter.add_vline(0, 0, q1, "black", "--", 2.0, f"Q1: {q1:.3f}")
    plotter.add_vline(0, 0, q3, "blue", "--", 2.0, f"Q3: {q3:.3f}")

    # histogram cosmetics
    plotter.add_grid(0, 0)
    plotter.add_oxy_labels(0, 0, "Value", "Frequency")
    plotter.add_legend(0, 0)

    # creating boxplot
    plotter.add_title(1, 0, f"Boxplot of '{point_of_interest}'")
    plotter.add_boxplot(current_column, vert=False, notch=True, irow=1, icol=0)
    plotter.add_oxy_labels(1, 0, xlabel="Value", ylabel="")
    plotter.add_grid(1, 0, axis="x", linestyle="--", alpha=0.3)

    # save plot if needed (validates dpi internally)
    if save:
        plotter.save_plot(save_path, dpi=dpi)
    # showing plot if needed
    if show:
        plotter.show()


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
        calc_statistics(
            point_of_interest=args.point_of_interest,
            show=args.show,
            save=args.save,
            save_path=args.save_path,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
        )
    except BaseException as val_err:
        print(f"ERROR while running main function:\n{val_err}")
