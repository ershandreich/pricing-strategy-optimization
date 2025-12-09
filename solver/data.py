import pandas as pd
from typing import Optional, Dict, List, Union, Literal
from pydantic import BaseModel


class ConstraintBlock(BaseModel):
    min_conversion_rate: Optional[float] = None
    min_completion_rate: Optional[float] = None
    min_take_rate: Optional[float] = None
    max_take_rate: Optional[float] = None
    min_rpi: Optional[float] = None
    min_icr: Optional[float] = None


class PricingConstraints(BaseModel):
    objective: Literal["rpi", "icr"] = "rpi"
    overall: Optional[ConstraintBlock] = ConstraintBlock()
    per_bin: Dict[str, ConstraintBlock] = {}


class DataManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.distance_vals = list(self.df["Distance"].unique())
        self.client_diffs = sorted(self.df["Client price diff"].unique())
        self.courier_diffs = sorted(self.df["Courier price diff"].unique())

        self.distance_to_idx = {d: i for i, d in enumerate(self.distance_vals)}
        self.client_diff_to_idx = {v: i for i, v in enumerate(self.client_diffs)}
        self.courier_diff_to_idx = {v: i for i, v in enumerate(self.courier_diffs)}

        self.n_bins = len(self.distance_vals)
        self.n_client = len(self.client_diffs)
        self.n_courier = len(self.courier_diffs)

        self.rpi: List[List[List[Union[None, float]]]] = [
            [[None] * self.n_courier for _ in range(self.n_client)]
            for _ in range(self.n_bins)
        ]

        self.icr: List[List[List[Union[None, float]]]] = [
            [[None] * self.n_courier for _ in range(self.n_client)]
            for _ in range(self.n_bins)
        ]

        self.client_conversion: List[List[Union[None, float]]] = [
            [None] * self.n_client
            for _ in range(self.n_bins)
        ]

        self.take_rate: List[List[List[Union[None, float]]]] = [
            [[None] * self.n_courier for _ in range(self.n_client)]
            for _ in range(self.n_bins)
        ]

        self.completion_rate: List[List[List[Union[None, float]]]] = [
            [[None] * self.n_courier for _ in range(self.n_client)]
            for _ in range(self.n_bins)
        ]

        self._fill_matrix()
        self._validate_full_matrix()

        self.distance_weights = self._extract_weights()

    def _fill_matrix(self):
        for _, row in self.df.iterrows():
            b = self.distance_to_idx[row["Distance"]]
            i = self.client_diff_to_idx[row["Client price diff"]]
            j = self.courier_diff_to_idx[row["Courier price diff"]]

            self.rpi[b][i][j] = row["Revenue per intent"]
            self.take_rate[b][i][j] = row["Take rate"]
            self.completion_rate[b][i][j] = row["Completion rate"]

            conv = row["Placed orders per intent"]
            if self.client_conversion[b][i] is None:
                self.client_conversion[b][i] = conv

            comp = row["Completed orders per intent"]
            self.icr[b][i][j] = comp

    def _validate_full_matrix(self):
        for b in range(self.n_bins):
            for i in range(self.n_client):
                if self.client_conversion[b][i] is None:
                    raise ValueError("Missing client conversion")
                for j in range(self.n_courier):
                    if self.rpi[b][i][j] is None:
                        raise ValueError("Missing revenue")
                    if self.take_rate[b][i][j] is None:
                        raise ValueError("Missing take rate")
                    if self.icr[b][i][j] is None:
                        raise ValueError("Missing completed conversion")
                    if self.completion_rate[b][i][j] is None:
                        raise ValueError("Missing completion rate")

    def _extract_weights(self):
        weights = []
        for d in self.distance_vals:
            subset = self.df[self.df["Distance"] == d]
            weight = subset["Distance share"].iloc[0]
            weights.append(weight)
        s = sum(weights)
        if abs(s - 1) > 1e-6:
            weights = [w / s for w in weights]
        return weights

    def calculate_strategy(self, strategy: dict) -> pd.Series:
        """
        strategy: dict mapping distance value -> (client_diff_value, courier_diff_value)
            Example:
            {
                "00 - 04 km": (0.0, 0.0),
                "04 - 08 km": (+0.1, -0.1),
                ...
            }

        Returns:
            pd.Series with the same metrics layout as in solver.prepare_solution() stats table
        """

        total_rpi = 0.0
        total_icr = 0.0
        overall_conv = 0.0
        overall_completed = 0.0
        overall_take_rate = 0.0
        overall_completion_rate = 0.0

        # per-bin (optional to return, but useful)
        take_rate_per_bin = {}
        completion_rate_per_bin = {}
        completed_conv_per_bin = {}

        for b, dist in enumerate(self.distance_vals):
            cdiff_val, kdiff_val = strategy[dist]

            i = self.client_diff_to_idx[cdiff_val]
            j = self.courier_diff_to_idx[kdiff_val]

            w = self.distance_weights[b]

            # add weighted metrics
            total_rpi += w * self.rpi[b][i][j]
            total_icr += w * self.icr[b][i][j]

            overall_conv += w * self.client_conversion[b][i]
            overall_completed += w * self.icr[b][i][j]
            overall_take_rate += w * self.take_rate[b][i][j]
            overall_completion_rate += w * self.completion_rate[b][i][j]

            # per-bin detail
            take_rate_per_bin[dist] = self.take_rate[b][i][j]
            completion_rate_per_bin[dist] = self.completion_rate[b][i][j]
            completed_conv_per_bin[dist] = self.icr[b][i][j]

        # return a single long Series (same structure as stats in prepare_solution)
        data = {
            "Total weighted revenue (RPI)": total_rpi,
            "Total weighted intent→completed (ICR)": total_icr,
            "Overall conversion rate": overall_conv,
            "Overall intent→completed rate": overall_completed,
            "Overall take rate": overall_take_rate,
            "Overall completion rate": overall_completion_rate,
        }

        # append per-bin metrics
        for dist in self.distance_vals:
            data[f"Take rate - {dist}"] = take_rate_per_bin[dist]
            data[f"Completion rate - {dist}"] = completion_rate_per_bin[dist]
            data[f"Intent→Completed rate - {dist}"] = completed_conv_per_bin[dist]

        return pd.Series(data)
