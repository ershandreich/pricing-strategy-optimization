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

        self.completion_rate: List[List[Union[None, float]]] = [
            [None] * self.n_courier
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

            conv = row["Placed orders per intent"]
            if self.client_conversion[b][i] is None:
                self.client_conversion[b][i] = conv

            comp = row["Completed orders per intent"]
            self.icr[b][i][j] = comp

            compl = row["Completion rate"]
            if self.completion_rate[b][j] is None:
                self.completion_rate[b][j] = compl

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
                    if self.completion_rate[b][j] is None:
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