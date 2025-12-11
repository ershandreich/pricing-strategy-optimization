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
        # Overall courier earnings constraint
        # ----------------------------------

        if (self.constraints.overall.min_courier_earnings is not None
                or self.constraints.overall.max_courier_earnings is not None):
            c_earnings = []
            for b in range(self.dm.n_bins):
                for i in range(self.dm.n_client):
                    for j in range(self.dm.n_courier):
                        c_earnings.append(self.dm.distance_weights[b] * self.dm.courier_payment[b][i][j] * self.x[b, i, j])

            if self.constraints.overall.min_courier_earnings is not None:
                self.solver.Add(sum(c_earnings) >= self.constraints.overall.min_courier_earnings)

            if self.constraints.overall.max_courier_earnings is not None:
                self.solver.Add(sum(c_earnings) <= self.constraints.overall.max_courier_earnings)

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
        overall_take_rate_opt = 0.0
        overall_completion_rate_opt = 0.0
        overall_courier_earnings_opt = 0.0

        # ----------------------------------
        # Optimal strategy rows
        # ----------------------------------
        for b in range(self.dm.n_bins):
            for i in range(self.dm.n_client):
                for j in range(self.dm.n_courier):

                    if self.x[b, i, j].solution_value() > 0.5:
                        dist = self.dm.distance_vals[b]
                        cdiff = self.dm.client_diffs[i]
                        kdiff = self.dm.courier_diffs[j]

                        rpi = self.dm.rpi[b][i][j]
                        icr = self.dm.icr[b][i][j]
                        conv = self.dm.client_conversion[b][i]
                        compl = self.dm.completion_rate[b][i][j]
                        take = self.dm.take_rate[b][i][j]
                        client_pay = self.dm.client_payment[b][i][j]
                        courier_pay = self.dm.courier_payment[b][i][j]

                        w = self.dm.distance_weights[b]

                        rows.append([
                            dist,
                            cdiff,
                            kdiff,
                            rpi,
                            icr,
                            conv,
                            compl,
                            take,
                            client_pay,
                            courier_pay
                        ])

                        total_rpi_opt += w * rpi
                        total_icr_opt += w * icr
                        overall_conv_opt += w * conv
                        overall_take_rate_opt += w * take
                        overall_completion_rate_opt += w * compl
                        overall_courier_earnings_opt += w * courier_pay

        df = pd.DataFrame(rows, columns=[
            "Distance",
            "Client price diff",
            "Courier price diff",
            "Revenue per intent",
            "Intent to completed rate",
            "Demand conversion rate",
            "Completion rate",
            "Take rate",
            "Client payment",
            "Courier payment"
        ])

        # ----------------------------------
        # Default strategy (0,0)
        # ----------------------------------
        zero_i = self.dm.client_diff_to_idx[0]
        zero_j = self.dm.courier_diff_to_idx[0]

        total_rpi_def = 0.0
        total_icr_def = 0.0
        overall_conv_def = 0.0
        overall_take_rate_def = 0.0
        overall_completion_rate_def = 0.0
        overall_courier_earnings_def = 0.0

        for b in range(self.dm.n_bins):
            w = self.dm.distance_weights[b]

            rpi_def = self.dm.rpi[b][zero_i][zero_j]
            icr_def = self.dm.icr[b][zero_i][zero_j]
            conv_def = self.dm.client_conversion[b][zero_i]
            compl_def = self.dm.completion_rate[b][zero_i][zero_j]
            take_def = self.dm.take_rate[b][zero_i][zero_j]
            courier_pay_def = self.dm.courier_payment[b][zero_i][zero_j]

            total_rpi_def += w * rpi_def
            total_icr_def += w * icr_def
            overall_conv_def += w * conv_def
            overall_take_rate_def += w * take_def
            overall_completion_rate_def += w * compl_def
            overall_courier_earnings_def += w * courier_pay_def

        # ----------------------------------
        # Stats table (ONLY overall metrics)
        # ----------------------------------
        stats = pd.DataFrame({
            "Optimal Strategy": [
                total_rpi_opt,
                total_icr_opt,
                overall_conv_opt,
                overall_take_rate_opt,
                overall_completion_rate_opt,
                overall_courier_earnings_opt
            ],
            "Default Strategy": [
                total_rpi_def,
                total_icr_def,
                overall_conv_def,
                overall_take_rate_def,
                overall_completion_rate_def,
                overall_courier_earnings_def
            ]
        }, index=[
            "Total weighted revenue (RPI)",
            "Total weighted intent→completed (ICR)",
            "Overall conversion rate",
            "Overall take rate",
            "Overall completion rate",
            "Overall courier earnings"
        ])

        # ----------------------------------
        # Percentage change
        # ----------------------------------
        pct = (stats["Optimal Strategy"] - stats["Default Strategy"]) / stats["Default Strategy"] * 100
        pct = pct.replace([float("inf"), -float("inf")], None)

        def fmt(x):
            if x is None or pd.isna(x):
                return None
            return f"{'+' if x > 0 else ''}{x:.2f}%"

        stats["% Change"] = pct.apply(fmt)

        return df, stats


