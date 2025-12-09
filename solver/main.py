from ortools.linear_solver import pywraplp
import pandas as pd

from .data import DataManager, PricingConstraints

class PricingStrategyOptimizer:
    def __init__(self, data_manager: DataManager, constraints: PricingConstraints):
        self.dm = data_manager
        self.constraints = constraints

    def _solve(self):
        self.solver = pywraplp.Solver.CreateSolver('CBC')
        if not self.solver:
            raise ValueError("Could not create solver")

        # ----------------------------------
        # Decision variables
        # ----------------------------------

        self.x = {}
        for b in range(self.dm.n_bins):
            for i in range(self.dm.n_client):
                for j in range(self.dm.n_courier):
                    self.x[b, i, j] = self.solver.BoolVar(f"x_b{b}_c{i}_k{j}")

        for b in range(self.dm.n_bins):
            self.solver.Add(sum(
                self.x[b, i, j]
                for i in range(self.dm.n_client)
                for j in range(self.dm.n_courier)
            ) == 1)

        # ----------------------------------
        # Overall completion rate constraint
        # ----------------------------------

        if self.constraints.overall.min_completion_rate is not None:
            min_comp = self.constraints.overall.min_completion_rate
            comps = []
            for b in range(self.dm.n_bins):
                for i in range(self.dm.n_client):
                    for j in range(self.dm.n_courier):
                        comps.append(
                            self.dm.distance_weights[b] *
                            self.dm.completion_rate[b][i][j] *
                            self.x[b, i, j]
                        )

            self.solver.Add(sum(comps) >= min_comp)

        # ----------------------------------
        # Overall conversion rate constraint
        # ----------------------------------

        if self.constraints.overall.min_conversion_rate is not None:
            min_conv = self.constraints.overall.min_conversion_rate
            convs = []
            for b in range(self.dm.n_bins):
                for i in range(self.dm.n_client):
                    for j in range(self.dm.n_courier):
                        convs.append(self.dm.distance_weights[b] * self.dm.client_conversion[b][i] * self.x[b, i, j])

            self.solver.Add(sum(convs) >= min_conv)

        # ----------------------------------
        # Overall take rate constraint
        # ----------------------------------

        if self.constraints.overall.min_take_rate is not None or self.constraints.overall.max_take_rate is not None:
            t_rates = []
            for b in range(self.dm.n_bins):
                for i in range(self.dm.n_client):
                    for j in range(self.dm.n_courier):
                        t_rates.append(self.dm.distance_weights[b] * self.dm.take_rate[b][i][j] * self.x[b, i, j])

            if self.constraints.overall.min_take_rate is not None:
                self.solver.Add(sum(t_rates) >= self.constraints.overall.min_take_rate)

            if self.constraints.overall.max_take_rate is not None:
                self.solver.Add(sum(t_rates) <= self.constraints.overall.max_take_rate)

        # ----------------------------------
        # Per-bin constraints
        # ----------------------------------
        if self.constraints and self.constraints.per_bin:

            for b in range(self.dm.n_bins):
                dist = self.dm.distance_vals[b]
                cb = self.constraints.per_bin.get(dist)
                if not cb:
                    continue

                # min conversion rate
                if cb.min_conversion_rate is not None:
                    M = self._metric_to_matrix(b, "conversion")
                    self._add_bin_constraint(b, M, cb.min_conversion_rate, op=">=")

                # min completion rate
                if cb.min_completion_rate is not None:
                    M = self._metric_to_matrix(b, "completion")
                    self._add_bin_constraint(b, M, cb.min_completion_rate, op=">=")

                # min take rate
                if cb.min_take_rate is not None:
                    M = self._metric_to_matrix(b, "take_rate")
                    self._add_bin_constraint(b, M, cb.min_take_rate, op=">=")

                # max take rate
                if cb.max_take_rate is not None:
                    M = self._metric_to_matrix(b, "take_rate")
                    self._add_bin_constraint(b, M, cb.max_take_rate, op="<=")

        # ----------------------------------
        # Guardrail constraint based on objective
        # ----------------------------------

        obj = self.constraints.objective if self.constraints else "rpi"
        overall = self.constraints.overall if self.constraints else None

        if overall:
            if obj == "rpi" and overall.min_icr is not None:
                icr_terms = []
                for b in range(self.dm.n_bins):
                    for i in range(self.dm.n_client):
                        for j in range(self.dm.n_courier):
                            icr_terms.append(
                                self.dm.distance_weights[b] *
                                self.dm.icr[b][i][j] *
                                self.x[b, i, j]
                            )
                self.solver.Add(sum(icr_terms) >= overall.min_icr)

            if obj == "icr" and overall.min_rpi is not None:
                rpi_terms = []
                for b in range(self.dm.n_bins):
                    for i in range(self.dm.n_client):
                        for j in range(self.dm.n_courier):
                            rpi_terms.append(
                                self.dm.distance_weights[b] *
                                self.dm.rpi[b][i][j] *
                                self.x[b, i, j]
                            )
                self.solver.Add(sum(rpi_terms) >= overall.min_rpi)

        # ----------------------------------
        # Objective function - either weighted revenue per intent (rpi) or intent to completed rate (icr)
        # ----------------------------------
        objective = self.solver.Objective()
        for b in range(self.dm.n_bins):
            for i in range(self.dm.n_client):
                for j in range(self.dm.n_courier):
                    if obj == "rpi":
                        value = self.dm.rpi[b][i][j]

                    elif obj == "icr":
                        value = self.dm.icr[b][i][j]

                    objective.SetCoefficient(self.x[b, i, j], self.dm.distance_weights[b] * value)

        objective.SetMaximization()

        status = self.solver.Solve()
        return status == pywraplp.Solver.OPTIMAL

    def _add_bin_constraint(self, b, metric_matrix, threshold, op=">="):
        expr = []
        for i in range(self.dm.n_client):
            for j in range(self.dm.n_courier):
                expr.append(metric_matrix[i][j] * self.x[b, i, j])

        lhs = sum(expr)

        if op == ">=":
            self.solver.Add(lhs >= threshold)
        else:
            self.solver.Add(lhs <= threshold)

    def _metric_to_matrix(self, b, metric_type):
        if metric_type == "conversion":
            # client_conversion[b][i] → expand to matrix[i][j]
            return [
                [self.dm.client_conversion[b][i] for j in range(self.dm.n_courier)]
                for i in range(self.dm.n_client)
            ]

        if metric_type == "completion":
            # already correct shape now
            return self.dm.completion_rate[b]

        if metric_type == "take_rate":
            # already matrix [i][j]
            return self.dm.take_rate[b]

    def solve(self):
        return self._solve()

    def prepare_solution(self):
        rows = []

        obj = self.constraints.objective if self.constraints else "rpi"

        total_rpi_opt = 0.0
        total_icr_opt = 0.0
        overall_conv_opt = 0.0
        overall_completed_opt = 0.0
        overall_take_rate_opt = 0.0
        overall_completion_rate_opt = 0.0

        # per-bin optimal metrics
        take_rate_opt_per_bin = [None] * self.dm.n_bins
        completion_rate_opt_per_bin = [None] * self.dm.n_bins
        completed_conv_opt_per_bin = [None] * self.dm.n_bins

        # ----------------------------------
        # Optimal strategy metrics
        # ----------------------------------
        for b in range(self.dm.n_bins):
            for i in range(self.dm.n_client):
                for j in range(self.dm.n_courier):
                    if self.x[b, i, j].solution_value() > 0.5:
                        dist = self.dm.distance_vals[b]
                        cdiff = self.dm.client_diffs[i]
                        kdiff = self.dm.courier_diffs[j]
                        raw_rev = self.dm.rpi[b][i][j]
                        w = self.dm.distance_weights[b]

                        rows.append([
                            dist,
                            cdiff,
                            kdiff,
                            raw_rev,
                            w,
                            raw_rev * w
                        ])

                        # weighted metrics
                        total_rpi_opt += w * self.dm.rpi[b][i][j]
                        total_icr_opt += w * self.dm.icr[b][i][j]

                        overall_conv_opt += w * self.dm.client_conversion[b][i]
                        overall_completed_opt += w * self.dm.icr[b][i][j]
                        overall_take_rate_opt += w * self.dm.take_rate[b][i][j]
                        overall_completion_rate_opt += w * self.dm.completion_rate[b][i][j]

                        # per-bin metrics
                        take_rate_opt_per_bin[b] = self.dm.take_rate[b][i][j]
                        completion_rate_opt_per_bin[b] = self.dm.completion_rate[b][i][j]
                        completed_conv_opt_per_bin[b] = self.dm.icr[b][i][j]

        df = pd.DataFrame(rows, columns=[
            "Distance",
            "Client price diff",
            "Courier price diff",
            "Revenue per intent",
            "Distance weight",
            "Weighted revenue"
        ])

        # ----------------------------------
        # Default strategy: CDiff=0, KDiff=0
        # ----------------------------------
        zero_i = self.dm.client_diff_to_idx[0]
        zero_j = self.dm.courier_diff_to_idx[0]

        total_rpi_def = 0.0
        total_icr_def = 0.0
        overall_conv_def = 0.0
        overall_completed_def = 0.0
        overall_take_rate_def = 0.0
        overall_completion_rate_def = 0.0

        # per-bin default metrics
        take_rate_def_per_bin = [None] * self.dm.n_bins
        completion_rate_def_per_bin = [None] * self.dm.n_bins
        completed_conv_def_per_bin = [None] * self.dm.n_bins

        for b in range(self.dm.n_bins):
            w = self.dm.distance_weights[b]

            total_rpi_def += w * self.dm.rpi[b][zero_i][zero_j]
            total_icr_def += w * self.dm.icr[b][zero_i][zero_j]

            overall_conv_def += w * self.dm.client_conversion[b][zero_i]
            overall_completed_def += w * self.dm.icr[b][zero_i][zero_j]
            overall_take_rate_def += w * self.dm.take_rate[b][zero_i][zero_j]
            overall_completion_rate_def += w * self.dm.completion_rate[b][zero_i][zero_j]

            take_rate_def_per_bin[b] = self.dm.take_rate[b][zero_i][zero_j]
            completion_rate_def_per_bin[b] = self.dm.completion_rate[b][zero_i][zero_j]
            completed_conv_def_per_bin[b] = self.dm.icr[b][zero_i][zero_j]

        # ----------------------------------
        # Stats table
        # ----------------------------------
        stats = pd.DataFrame({
            "Optimal Strategy": [
                total_rpi_opt,
                total_icr_opt,
                overall_conv_opt,
                overall_completed_opt,
                overall_take_rate_opt,
                overall_completion_rate_opt
            ],
            "Default Strategy": [
                total_rpi_def,
                total_icr_def,
                overall_conv_def,
                overall_completed_def,
                overall_take_rate_def,
                overall_completion_rate_def
            ]
        }, index=[
            "Total weighted revenue (RPI)",
            "Total weighted intent→completed (ICR)",
            "Overall conversion rate",
            "Overall intent→completed rate",
            "Overall take rate",
            "Overall completion rate"
        ])

        # ----------------------------------
        # Add per-bin metrics in metric-first grouping
        # ----------------------------------

        # 1. Take rate per bin
        for b in range(self.dm.n_bins):
            dist_label = self.dm.distance_vals[b]
            stats.loc[f"Take rate - {dist_label}"] = [
                take_rate_opt_per_bin[b],
                take_rate_def_per_bin[b]
            ]

        # 2. Completion rate per bin
        for b in range(self.dm.n_bins):
            dist_label = self.dm.distance_vals[b]
            stats.loc[f"Completion rate - {dist_label}"] = [
                completion_rate_opt_per_bin[b],
                completion_rate_def_per_bin[b]
            ]

        # 3. Intent→Completed rate per bin
        for b in range(self.dm.n_bins):
            dist_label = self.dm.distance_vals[b]
            stats.loc[f"Intent→Completed rate - {dist_label}"] = [
                completed_conv_opt_per_bin[b],
                completed_conv_def_per_bin[b]
            ]

        return df, stats