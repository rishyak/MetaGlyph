"""Stage 6: Aggregation & Reporting.

This module aggregates evaluation results and produces summary statistics,
tables, and figures for the paper.

Aggregation dimensions:
- Task family
- Operator
- Prompt condition (NL, MG, CTRL)
- Model

Artifacts produced:
- summary/tables/*.csv      - Result tables
- summary/figures/*.pdf     - Visualizations
- summary/stats.json        - Statistical summary
"""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
import json

from ..utils.io_utils import load_json, save_json, ensure_dir, list_files


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a group."""
    group_key: str
    group_value: str
    count: int
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    exact_match_rate: float
    parse_success_rate: float
    overall_pass_rate: float
    semantic_equivalence_rate: float = 0.0  # NL==MG and both!=CTRL

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenStats:
    """Token statistics for prompts."""
    family: str
    model: str
    nl_instruction_tokens: float
    mg_instruction_tokens: float
    ctrl_instruction_tokens: float
    nl_total_tokens: float
    mg_total_tokens: float
    ctrl_total_tokens: float
    instruction_reduction_vs_nl: float  # (NL - MG) / NL for instructions only
    full_prompt_reduction_vs_nl: float  # (NL - MG) / NL for full prompts

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OperatorFidelityStats:
    """Statistics for operator fidelity."""
    operator: str
    check_count: int
    pass_count: int
    pass_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConditionComparison:
    """Comparison between conditions (NL, MG, CTRL)."""
    family: str
    model: str
    nl_accuracy: float
    mg_accuracy: float
    ctrl_accuracy: float
    mg_vs_nl_diff: float
    mg_vs_ctrl_diff: float
    statistical_significant: bool

    def to_dict(self) -> dict:
        return asdict(self)


class StatisticalTests:
    """Statistical testing utilities."""

    @staticmethod
    def mean(values: list[float]) -> float:
        """Calculate mean."""
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def std(values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean_val = StatisticalTests.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    @staticmethod
    def paired_t_test(values1: list[float], values2: list[float], alpha: float = 0.05) -> tuple[float, bool]:
        """
        Perform paired t-test.

        Returns (t_statistic, is_significant)
        """
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0, False

        # Calculate differences
        diffs = [v1 - v2 for v1, v2 in zip(values1, values2)]
        n = len(diffs)

        # Mean and std of differences
        mean_diff = StatisticalTests.mean(diffs)
        std_diff = StatisticalTests.std(diffs)

        if std_diff == 0:
            return float('inf') if mean_diff != 0 else 0.0, mean_diff != 0

        # t-statistic
        t_stat = mean_diff / (std_diff / (n ** 0.5))

        # Critical value (two-tailed, approximate)
        # For n >= 30, use z-value; otherwise use t-table approximation
        critical = 1.96 if n >= 30 else 2.0 + (4.0 / n)

        return t_stat, abs(t_stat) > critical


class TableGenerator:
    """Generates CSV tables from aggregated results."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def generate_accuracy_table(
        self,
        metrics_by_family: dict[str, dict[str, AggregatedMetrics]],
        filename: str = "accuracy_by_family_condition.csv",
    ) -> Path:
        """Generate accuracy table by family and condition."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Family", "Condition", "N", "Accuracy (Mean)", "Accuracy (Std)",
                "F1 (Mean)", "Exact Match Rate", "Parse Rate", "Pass Rate", "Semantic Equiv Rate"
            ])

            # Data rows
            for family, conditions in sorted(metrics_by_family.items()):
                for condition, metrics in sorted(conditions.items()):
                    writer.writerow([
                        family,
                        condition,
                        metrics.count,
                        f"{metrics.accuracy_mean:.3f}",
                        f"{metrics.accuracy_std:.3f}",
                        f"{metrics.f1_mean:.3f}",
                        f"{metrics.exact_match_rate:.3f}",
                        f"{metrics.parse_success_rate:.3f}",
                        f"{metrics.overall_pass_rate:.3f}",
                        f"{metrics.semantic_equivalence_rate:.3f}",
                    ])

        return filepath

    def generate_operator_fidelity_table(
        self,
        fidelity_stats: list[OperatorFidelityStats],
        filename: str = "operator_fidelity.csv",
    ) -> Path:
        """Generate operator fidelity table."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["Operator", "Checks", "Passes", "Pass Rate"])

            # Data rows
            for stats in sorted(fidelity_stats, key=lambda x: x.operator):
                writer.writerow([
                    stats.operator,
                    stats.check_count,
                    stats.pass_count,
                    f"{stats.pass_rate:.3f}",
                ])

        return filepath

    def generate_comparison_table(
        self,
        comparisons: list[ConditionComparison],
        filename: str = "condition_comparison.csv",
    ) -> Path:
        """Generate NL vs MG vs CTRL comparison table."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Family", "Model", "NL Accuracy", "MG Accuracy", "CTRL Accuracy",
                "MG-NL Diff", "MG-CTRL Diff", "Significant"
            ])

            # Data rows
            for comp in comparisons:
                writer.writerow([
                    comp.family,
                    comp.model,
                    f"{comp.nl_accuracy:.3f}",
                    f"{comp.mg_accuracy:.3f}",
                    f"{comp.ctrl_accuracy:.3f}",
                    f"{comp.mg_vs_nl_diff:+.3f}",
                    f"{comp.mg_vs_ctrl_diff:+.3f}",
                    "Yes" if comp.statistical_significant else "No",
                ])

        return filepath

    def generate_token_compression_table(
        self,
        compression_stats: list[dict],
        filename: str = "token_compression.csv",
    ) -> Path:
        """Generate token compression statistics table."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Family",
                "Avg NL Instruction Tokens", "Avg MG Instruction Tokens",
                "Avg NL Total Tokens", "Avg MG Total Tokens",
                "Instruction Compression Ratio", "Full Prompt Compression Ratio",
                "Instruction Savings (%)", "Full Prompt Savings (%)"
            ])

            # Data rows
            for stats in compression_stats:
                nl_tokens = stats.get("avg_nl_instruction_tokens", 0)
                mg_tokens = stats.get("avg_mg_instruction_tokens", 0)
                nl_total = stats.get("avg_nl_total_tokens", 0)
                mg_total = stats.get("avg_mg_total_tokens", 0)
                instruction_ratio = stats.get("instruction_compression_ratio", 0)
                full_ratio = stats.get("full_prompt_compression_ratio", 0)
                instruction_savings = (1 - instruction_ratio) * 100 if instruction_ratio > 0 else 0
                full_savings = (1 - full_ratio) * 100 if full_ratio > 0 else 0

                writer.writerow([
                    stats.get("family", ""),
                    f"{nl_tokens:.1f}",
                    f"{mg_tokens:.1f}",
                    f"{nl_total:.1f}",
                    f"{mg_total:.1f}",
                    f"{instruction_ratio:.3f}",
                    f"{full_ratio:.3f}",
                    f"{instruction_savings:.1f}%",
                    f"{full_savings:.1f}%",
                ])

        return filepath

    def generate_token_comparison_table(
        self,
        token_stats: list[TokenStats],
        filename: str = "token_comparison.csv",
    ) -> Path:
        """Generate token comparison table showing NL, MG, CTRL tokens per model."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Model", "Family",
                "NL Instruction Tokens", "MG Instruction Tokens", "CTRL Instruction Tokens",
                "NL Total Tokens", "MG Total Tokens", "CTRL Total Tokens",
                "Instruction Reduction vs NL", "Full Prompt Reduction vs NL"
            ])

            # Data rows
            for stats in token_stats:
                writer.writerow([
                    stats.model,
                    stats.family,
                    f"{stats.nl_instruction_tokens:.1f}",
                    f"{stats.mg_instruction_tokens:.1f}",
                    f"{stats.ctrl_instruction_tokens:.1f}",
                    f"{stats.nl_total_tokens:.1f}",
                    f"{stats.mg_total_tokens:.1f}",
                    f"{stats.ctrl_total_tokens:.1f}",
                    f"{stats.instruction_reduction_vs_nl:.1%}",
                    f"{stats.full_prompt_reduction_vs_nl:.1%}",
                ])

        return filepath


class FigureGenerator:
    """Generates visualizations from aggregated results."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def generate_accuracy_bar_chart(
        self,
        metrics_by_family: dict[str, dict[str, AggregatedMetrics]],
        filename: str = "accuracy_comparison.pdf",
    ) -> Optional[Path]:
        """Generate accuracy comparison bar chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        filepath = self.output_dir / filename

        families = sorted(metrics_by_family.keys())
        preferred = ["NL", "NL_SHORT", "ASCII_DSL", "MG", "CTRL", "CTRL_RANDOM"]
        present = {
            condition
            for family_conditions in metrics_by_family.values()
            for condition in family_conditions.keys()
        }
        conditions = [condition for condition in preferred if condition in present]
        conditions.extend(sorted(present - set(conditions)))

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(families))
        width = 0.8 / max(len(conditions), 1)

        for i, condition in enumerate(conditions):
            accuracies = []
            errors = []
            for family in families:
                if condition in metrics_by_family[family]:
                    m = metrics_by_family[family][condition]
                    accuracies.append(m.accuracy_mean)
                    errors.append(m.accuracy_std)
                else:
                    accuracies.append(0)
                    errors.append(0)

            bars = ax.bar(x + i * width, accuracies, width, label=condition, yerr=errors, capsize=3)

        ax.set_xlabel('Task Family')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Task Family and Instruction Condition')
        ax.set_xticks(x + (width * (len(conditions) - 1) / 2))
        ax.set_xticklabels([f.replace('_', '\n') for f in families], fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def generate_operator_fidelity_chart(
        self,
        fidelity_stats: list[OperatorFidelityStats],
        filename: str = "operator_fidelity.pdf",
    ) -> Optional[Path]:
        """Generate operator fidelity bar chart."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        filepath = self.output_dir / filename

        operators = [s.operator for s in fidelity_stats]
        pass_rates = [s.pass_rate for s in fidelity_stats]

        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.bar(operators, pass_rates, color='steelblue')

        ax.set_xlabel('Operator')
        ax.set_ylabel('Pass Rate')
        ax.set_title('Operator Fidelity Pass Rates')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')

        # Add value labels on bars
        for bar, rate in zip(bars, pass_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return filepath


class Aggregator:
    """Main class for aggregating results and generating reports."""

    def __init__(
        self,
        results_dir: Path | str,
        tokens_dir: Path | str,
        summary_dir: Path | str,
        models: list[str],
        prompts_dir: Optional[Path | str] = None,
        tokenizer_model: str = "simple",
    ):
        self.results_dir = Path(results_dir)
        self.tokens_dir = Path(tokens_dir)
        self.summary_dir = Path(summary_dir)
        self.models = models
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.tokenizer_model = tokenizer_model

        self.stats = StatisticalTests()
        self.table_gen = TableGenerator(self.summary_dir / "tables")
        self.figure_gen = FigureGenerator(self.summary_dir / "figures")

    def aggregate_all(self) -> dict[str, Any]:
        """Run all aggregations and generate all reports."""
        summary = {}
        all_token_stats = []

        for model in self.models:
            model_results = self._load_model_results(model)
            if not model_results:
                continue

            # Aggregate by family and condition
            metrics_by_family = self._aggregate_by_family_condition(model_results)
            summary[f"{model}_metrics"] = {
                family: {cond: m.to_dict() for cond, m in conditions.items()}
                for family, conditions in metrics_by_family.items()
            }

            # Aggregate operator fidelity
            fidelity_stats = self._aggregate_operator_fidelity(model_results)
            summary[f"{model}_fidelity"] = [s.to_dict() for s in fidelity_stats]

            # Generate comparisons
            comparisons = self._generate_comparisons(model, model_results)
            summary[f"{model}_comparisons"] = [c.to_dict() for c in comparisons]

            # Calculate token stats if prompts_dir is available
            if self.prompts_dir:
                token_stats = self._calculate_token_stats(model)
                all_token_stats.extend(token_stats)
                summary[f"{model}_tokens"] = [s.to_dict() for s in token_stats]

            # Generate tables
            self.table_gen.generate_accuracy_table(metrics_by_family, f"accuracy_{model}.csv")
            self.table_gen.generate_operator_fidelity_table(fidelity_stats, f"fidelity_{model}.csv")
            self.table_gen.generate_comparison_table(comparisons, f"comparison_{model}.csv")

            # Generate figures
            self.figure_gen.generate_accuracy_bar_chart(metrics_by_family, f"accuracy_{model}.pdf")
            self.figure_gen.generate_operator_fidelity_chart(fidelity_stats, f"fidelity_{model}.pdf")

        # Generate combined token comparison table
        if all_token_stats:
            self.table_gen.generate_token_comparison_table(all_token_stats, "token_comparison.csv")

        # Save summary JSON
        save_json(summary, self.summary_dir / "stats.json")

        return summary

    def _calculate_token_stats(self, model: str) -> list[TokenStats]:
        """Calculate token statistics for each family from Stage 3 outputs."""
        token_stats = []

        model_tokens_dir = self.tokens_dir / model
        if not model_tokens_dir.exists():
            return token_stats

        for family_dir in model_tokens_dir.iterdir():
            if not family_dir.is_dir():
                continue

            family_name = family_dir.name
            by_condition = defaultdict(lambda: {"instruction": [], "total": []})

            for token_file in list_files(family_dir, "*.json"):
                counts = load_json(token_file)
                condition = counts.get("condition") or counts.get("prompt_id", "").rsplit("_", 1)[-1]
                by_condition[condition]["instruction"].append(counts.get("instruction_tokens", 0))
                by_condition[condition]["total"].append(counts.get("total_tokens", 0))

            def avg(values: list[float]) -> float:
                return sum(values) / len(values) if values else 0.0

            nl_instruction = avg(by_condition["NL"]["instruction"])
            mg_instruction = avg(by_condition["MG"]["instruction"])
            ctrl_instruction = avg(by_condition["CTRL"]["instruction"])
            nl_total = avg(by_condition["NL"]["total"])
            mg_total = avg(by_condition["MG"]["total"])
            ctrl_total = avg(by_condition["CTRL"]["total"])

            instruction_reduction = (
                (nl_instruction - mg_instruction) / nl_instruction
                if nl_instruction > 0 else 0
            )
            full_prompt_reduction = (
                (nl_total - mg_total) / nl_total
                if nl_total > 0 else 0
            )

            token_stats.append(TokenStats(
                family=family_name,
                model=model,
                nl_instruction_tokens=nl_instruction,
                mg_instruction_tokens=mg_instruction,
                ctrl_instruction_tokens=ctrl_instruction,
                nl_total_tokens=nl_total,
                mg_total_tokens=mg_total,
                ctrl_total_tokens=ctrl_total,
                instruction_reduction_vs_nl=instruction_reduction,
                full_prompt_reduction_vs_nl=full_prompt_reduction,
            ))

        return token_stats

    def _load_model_results(self, model: str) -> list[dict]:
        """Load all evaluation results for a model."""
        results = []
        model_dir = self.results_dir / model

        if not model_dir.exists():
            return results

        # Results are now in: results/<model>/<family>/*.json
        for family_dir in model_dir.iterdir():
            if not family_dir.is_dir():
                continue

            for result_file in list_files(family_dir, "*.json"):
                result = load_json(result_file)
                results.append(result)

        return results

    def _aggregate_by_family_condition(
        self,
        results: list[dict],
    ) -> dict[str, dict[str, AggregatedMetrics]]:
        """Aggregate results by task family and condition."""
        # Group results
        groups = defaultdict(lambda: defaultdict(list))
        for r in results:
            family = r.get("family", "unknown")
            condition = r.get("condition", "unknown")
            groups[family][condition].append(r)

        # Calculate metrics
        aggregated = {}
        for family, conditions in groups.items():
            aggregated[family] = {}
            for condition, condition_results in conditions.items():
                metrics = self._calculate_group_metrics(
                    group_key="condition",
                    group_value=condition,
                    results=condition_results,
                )
                aggregated[family][condition] = metrics

        return aggregated

    def _calculate_group_metrics(
        self,
        group_key: str,
        group_value: str,
        results: list[dict],
    ) -> AggregatedMetrics:
        """Calculate aggregated metrics for a group of results."""
        n = len(results)
        if n == 0:
            return AggregatedMetrics(
                group_key=group_key,
                group_value=group_value,
                count=0,
                accuracy_mean=0.0,
                accuracy_std=0.0,
                f1_mean=0.0,
                f1_std=0.0,
                exact_match_rate=0.0,
                parse_success_rate=0.0,
                overall_pass_rate=0.0,
            )

        accuracies = [r.get("accuracy", 0) for r in results]
        f1_scores = [r.get("f1_score", 0) for r in results]
        exact_matches = [1 if r.get("exact_match", False) else 0 for r in results]
        parse_successes = [1 if r.get("parse_success", False) else 0 for r in results]
        overall_passes = [1 if r.get("overall_pass", False) else 0 for r in results]
        semantic_equiv = [1 if r.get("semantic_equivalence_pass", False) else 0 for r in results]

        return AggregatedMetrics(
            group_key=group_key,
            group_value=group_value,
            count=n,
            accuracy_mean=self.stats.mean(accuracies),
            accuracy_std=self.stats.std(accuracies),
            f1_mean=self.stats.mean(f1_scores),
            f1_std=self.stats.std(f1_scores),
            exact_match_rate=self.stats.mean(exact_matches),
            parse_success_rate=self.stats.mean(parse_successes),
            overall_pass_rate=self.stats.mean(overall_passes),
            semantic_equivalence_rate=self.stats.mean(semantic_equiv),
        )

    def _aggregate_operator_fidelity(
        self,
        results: list[dict],
    ) -> list[OperatorFidelityStats]:
        """Aggregate operator fidelity statistics."""
        operator_stats = defaultdict(lambda: {"checks": 0, "passes": 0})

        for r in results:
            for fidelity in r.get("operator_fidelity", []):
                op = fidelity.get("operator", "")
                if fidelity.get("checked", False):
                    operator_stats[op]["checks"] += 1
                    if fidelity.get("passed", False):
                        operator_stats[op]["passes"] += 1

        stats = []
        for op, counts in operator_stats.items():
            checks = counts["checks"]
            passes = counts["passes"]
            stats.append(OperatorFidelityStats(
                operator=op,
                check_count=checks,
                pass_count=passes,
                pass_rate=passes / checks if checks > 0 else 0.0,
            ))

        return stats

    def _generate_comparisons(
        self,
        model: str,
        results: list[dict],
    ) -> list[ConditionComparison]:
        """Generate NL vs MG vs CTRL comparisons."""
        # Group by family
        by_family = defaultdict(lambda: defaultdict(list))
        for r in results:
            family = r.get("family", "")
            condition = r.get("condition", "")
            by_family[family][condition].append(r.get("accuracy", 0))

        comparisons = []
        for family, conditions in by_family.items():
            nl_acc = conditions.get("NL", [])
            mg_acc = conditions.get("MG", [])
            ctrl_acc = conditions.get("CTRL", [])

            if not (nl_acc and mg_acc and ctrl_acc):
                continue

            nl_mean = self.stats.mean(nl_acc)
            mg_mean = self.stats.mean(mg_acc)
            ctrl_mean = self.stats.mean(ctrl_acc)

            # Test significance of MG vs NL
            _, significant = self.stats.paired_t_test(mg_acc, nl_acc)

            comparisons.append(ConditionComparison(
                family=family,
                model=model,
                nl_accuracy=nl_mean,
                mg_accuracy=mg_mean,
                ctrl_accuracy=ctrl_mean,
                mg_vs_nl_diff=mg_mean - nl_mean,
                mg_vs_ctrl_diff=mg_mean - ctrl_mean,
                statistical_significant=significant,
            ))

        return comparisons

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get high-level summary statistics."""
        summary_path = self.summary_dir / "stats.json"
        if summary_path.exists():
            return load_json(summary_path)
        return {}
