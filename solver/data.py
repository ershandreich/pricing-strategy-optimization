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


class RawDataProcessor:
    REQUIRED_SHEETS = ["Client data", "Courier data", "Weights"]

    CLIENT_COLUMNS = [
        "Distance bin",
        "Client price diff",
        "Demand conversion rate",
        "Client payment"
    ]

    COURIER_COLUMNS = [
        "Distance bin",
        "Courier price diff",
        "Completion rate",
        "Courier payment"
    ]

    WEIGHTS_COLUMNS = [
        "Distance bin",
        "Weight"
    ]

    def __init__(self, file):
        self.file = file

        self.df_client = None
        self.df_courier = None
        self.df_weights = None

        self.df_merged = None

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def process(self):
        self._load()
        self._validate()
        self._merge()
        return self.df_merged

    # -----------------------------
    # INTERNAL STEPS
    # -----------------------------
    def _load(self):
        try:
            xls = pd.ExcelFile(self.file)
        except Exception as e:
            raise ValueError(f"Cannot read Excel file: {e}")

        # check required sheets
        missing = [s for s in self.REQUIRED_SHEETS if s not in xls.sheet_names]
        if missing:
            raise ValueError(f"Missing required sheets: {missing}")

        # load each sheet
        self.df_client = pd.read_excel(self.file, sheet_name="Client data")
        self.df_courier = pd.read_excel(self.file, sheet_name="Courier data")
        self.df_weights = pd.read_excel(self.file, sheet_name="Weights")

    # -----------------------------
    # VALIDATION
    # -----------------------------
    def _validate(self):
        self._validate_columns()
        self._validate_distance_bins()
        self._validate_price_diff_combinations()
        self._validate_weights()

    def _validate_columns(self):
        # client
        missing_client = [c for c in self.CLIENT_COLUMNS if c not in self.df_client.columns]
        if missing_client:
            raise ValueError(f"Client data missing columns: {missing_client}")

        # courier
        missing_courier = [c for c in self.COURIER_COLUMNS if c not in self.df_courier.columns]
        if missing_courier:
            raise ValueError(f"Courier data missing columns: {missing_courier}")

        # weights
        missing_weights = [c for c in self.WEIGHTS_COLUMNS if c not in self.df_weights.columns]
        if missing_weights:
            raise ValueError(f"Weights data missing columns: {missing_weights}")

    def _validate_distance_bins(self):
        client_bins = set(self.df_client["Distance bin"])
        courier_bins = set(self.df_courier["Distance bin"])
        weight_bins = set(self.df_weights["Distance bin"])

        # courier must contain all client bins
        missing = client_bins - courier_bins
        if missing:
            raise ValueError(f"Courier data missing distance bins: {missing}")

        # weights must exactly match client bins
        if client_bins != weight_bins:
            raise ValueError(
                f"Weights distance bins do not match client bins.\n"
                f"Client: {client_bins}\nWeights: {weight_bins}"
            )

    def _validate_price_diff_combinations(self):
        # client diffs
        client_counts = (
            self.df_client.groupby("Distance bin")["Client price diff"].nunique()
        )

        if client_counts.nunique() != 1:
            raise ValueError(
                f"Client data has inconsistent number of price diffs across distance bins: {client_counts.to_dict()}"
            )

        # courier diffs
        courier_counts = (
            self.df_courier.groupby("Distance bin")["Courier price diff"].nunique()
        )

        if courier_counts.nunique() != 1:
            raise ValueError(
                f"Courier data has inconsistent number of price diffs across distance bins: {courier_counts.to_dict()}"
            )

    def _validate_weights(self):
        # sum = 1
        wsum = self.df_weights["Weight"].sum()
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1. Current sum: {wsum}")

        # all unique
        if self.df_weights["Distance bin"].duplicated().any():
            raise ValueError("Weights table contains duplicate distance bins.")

    def _merge(self):
        """
        Build scenario table:
        - For each distance bin combine every client diff with every courier diff
        - Add weight
        - Compute Take rate, ICR, RPI
        """

        rows = []

        # dictionary for quick weight lookup
        weight_map = dict(zip(self.df_weights["Distance bin"], self.df_weights["Weight"]))

        # iterate by distance bin
        for dbin in self.df_client["Distance bin"].unique():

            dfc = self.df_client[self.df_client["Distance bin"] == dbin]
            dfk = self.df_courier[self.df_courier["Distance bin"] == dbin]
            w = weight_map[dbin]

            # cartesian product of client and courier diffs
            for _, c_row in dfc.iterrows():
                for _, k_row in dfk.iterrows():
                    client_payment = float(c_row["Client payment"])
                    courier_payment = float(k_row["Courier payment"])

                    demand_conv = float(c_row["Demand conversion rate"])
                    completion_rate = float(k_row["Completion rate"])

                    # --- compute metrics ---
                    take_rate = (client_payment - courier_payment) / client_payment
                    icr = demand_conv * completion_rate
                    rpi = client_payment * take_rate * icr

                    rows.append({
                        "Distance bin": dbin,
                        "Client price diff": c_row["Client price diff"],
                        "Courier price diff": k_row["Courier price diff"],

                        "Demand conversion rate": demand_conv,
                        "Client payment": client_payment,
                        "Completion rate": completion_rate,
                        "Courier payment": courier_payment,
                        "Weight": w,

                        # computed metrics
                        "Take rate": take_rate,
                        "ICR": icr,
                        "RPI": rpi,
                    })

        self.df_merged = pd.DataFrame(rows)


class DataManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.distance_vals = list(self.df["Distance bin"].unique())
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
            b = self.distance_to_idx[row["Distance bin"]]
            i = self.client_diff_to_idx[row["Client price diff"]]
            j = self.courier_diff_to_idx[row["Courier price diff"]]

            self.rpi[b][i][j] = row["RPI"]
            self.icr[b][i][j] = row["ICR"]
            self.take_rate[b][i][j] = row["Take rate"]
            self.completion_rate[b][i][j] = row["Completion rate"]
            self.client_conversion[b][i] = row["Demand conversion rate"]


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
            subset = self.df[self.df["Distance bin"] == d]
            weight = subset["Weight"].iloc[0]
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
