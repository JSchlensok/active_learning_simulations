from __future__ import annotations

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Optional
from biocentral_api import SequenceData, ActiveLearningCampaignConfig, ActiveLearningSimulationConfig, \
    ActiveLearningOptimizationMode, ActiveLearningModelType, BiocentralAPI, ActiveLearningIterationResult, \
    ActiveLearningSimulationResult, ActiveLearningConvergenceConfig

from pydantic import BaseModel


class ActiveLearningFixedBaseConfig(BaseModel):
    class Config:
        frozen = False

    """ Fixed base config for active learning simulations that does not change throughout simulation"""
    # Simulation config
    simulation_data: List[SequenceData]

    # Campaign config
    optimization_mode: ActiveLearningOptimizationMode
    target_lb: Optional[float] = None
    target_ub: Optional[float] = None
    target_value: Optional[float] = None
    discrete_targets: Optional[List[str]] = None


class ActiveLearningSimulator:
    def __init__(self, al_base_config: ActiveLearningFixedBaseConfig):
        self.base_config = al_base_config
        self.biocentral_api = BiocentralAPI(local_only=True)

    def _run_simulation(self, model_type: ActiveLearningModelType, embedder_name: str,
                        seed: int) -> ActiveLearningSingleSimulationResult:
        al_campaign_config = ActiveLearningCampaignConfig(name="Test",  # TODO
                                                          embedder_name=embedder_name,
                                                          model_type=model_type,
                                                          optimization_mode=self.base_config.optimization_mode,
                                                          seed=seed,
                                                          target_lb=self.base_config.target_lb,
                                                          target_ub=self.base_config.target_ub,
                                                          target_value=self.base_config.target_value,
                                                          discrete_targets=self.base_config.discrete_targets)
        al_simulation_config = ActiveLearningSimulationConfig(simulation_data=self.base_config.simulation_data,
                                                              n_start=10,  # TODO
                                                              n_suggestions_per_iteration=5,  # TODO
                                                              convergence_config=ActiveLearningConvergenceConfig(
                                                                  max_labels_budget=50,
                                                                  target_successes=10,
                                                                  max_consecutive_failures=5
                                                              ),  # TODO
                                                              )
        result = self.biocentral_api.al_simulation(campaign_config=al_campaign_config,
                                                   simulation_config=al_simulation_config).run_with_progress()
        return ActiveLearningSingleSimulationResult(al_campaign_config=al_campaign_config,
                                                    al_simulation_config=al_simulation_config,
                                                    simulation_result=result)

    def simulate(self, embedder_name: str, model_type: ActiveLearningModelType,
                 n_rounds: int) -> ActiveLearningMultipleSimulationResult:
        simulation_results = []
        for iteration_idx in range(n_rounds):
            print(f"Running simulation round {iteration_idx + 1}/{n_rounds}...")
            seed = 42 + iteration_idx
            single_simulation_result = self._run_simulation(model_type=model_type,
                                                            embedder_name=embedder_name,
                                                            seed=seed)
            simulation_results.append(single_simulation_result)
        return ActiveLearningMultipleSimulationResult(simulation_results)


class ActiveLearningSingleSimulationResult:
    def __init__(self, al_campaign_config: ActiveLearningCampaignConfig,
                 al_simulation_config: ActiveLearningSimulationConfig,
                 simulation_result: ActiveLearningSimulationResult):
        self.al_campaign_config = al_campaign_config
        self.al_simulation_config = al_simulation_config
        self.simulation_result = simulation_result

    def model_dump_json(self):
        """Serialize to JSON string"""
        return json.dumps({
            'al_campaign_config': json.loads(self.al_campaign_config.model_dump_json()),
            'al_simulation_config': json.loads(self.al_simulation_config.model_dump_json()),
            'simulation_result': json.loads(self.simulation_result.model_dump_json())
        })

    @classmethod
    def model_validate_json(cls, json_str: str):
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls(
            al_campaign_config=ActiveLearningCampaignConfig.model_validate_json(json.dumps(data['al_campaign_config'])),
            al_simulation_config=ActiveLearningSimulationConfig.model_validate_json(
                json.dumps(data['al_simulation_config'])),
            simulation_result=ActiveLearningSimulationResult.model_validate_json(json.dumps(data['simulation_result']))
        )

    def is_success(self):
        required_target_successes = 10  # TODO CONFIG
        return sum(self.simulation_result.iteration_target_successes) >= required_target_successes

    def _visualize_results(self):
        """Visualize active learning simulation results"""
        result = self.simulation_result
        is_discrete = self.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE

        metric_name = "Accuracy" if is_discrete else "RMSE"

        # Set style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 12))

        # --- ensure all iteration-based series share a single length ---
        n_rounds_candidates = [
            len(getattr(result, "iteration_results", []) or []),
            len(getattr(result, "iteration_metrics_total", []) or []),
            len(getattr(result, "iteration_metrics_suggestions", []) or []),
            len(getattr(result, "iteration_target_successes", []) or []),
            len(getattr(result, "iteration_consecutive_failures", []) or []),
        ]
        n_rounds = max(n_rounds_candidates) if any(n_rounds_candidates) else 0

        iteration_metrics_total = [m.mean for m in (result.iteration_metrics_total or [])[:n_rounds]]
        iteration_metrics_suggestions = [m.mean for m in (result.iteration_metrics_suggestions or [])[:n_rounds]]
        iteration_target_successes = (result.iteration_target_successes or [])[:n_rounds]
        iteration_consecutive_failures = (result.iteration_consecutive_failures or [])[:n_rounds]
        iteration_results = (result.iteration_results or [])[:n_rounds]

        iterations = list(range(1, n_rounds + 1))
        # --- end alignment ---

        # 1. Metrics Evolution (Total vs Suggestions)
        ax1 = plt.subplot(3, 3, 1)
        if iteration_metrics_total:
            plt.plot(iterations[:len(iteration_metrics_total)], iteration_metrics_total, marker='o', label='All Data',
                     linewidth=2)
        if iteration_metrics_suggestions:
            plt.plot(iterations[:len(iteration_metrics_suggestions)], iteration_metrics_suggestions, marker='s',
                     label='Suggestions', linewidth=2)

        # Overlay success markers (use aligned lists)
        success_iterations = [i + 1 for i, s in enumerate(iteration_target_successes) if s > 0]
        if success_iterations and iteration_metrics_total:
            success_metrics = [iteration_metrics_total[i - 1] for i in success_iterations if
                               (i - 1) < len(iteration_metrics_total)]
            plt.scatter(success_iterations[:len(success_metrics)], success_metrics, color='gold', s=200,
                        marker='*', edgecolors='black', linewidths=1.5,
                        label='Target Found', zorder=5)

        plt.title('Metric Comparison with Success Markers')
        plt.xlabel('Iteration')
        plt.ylabel(f'Metric ({metric_name})')
        plt.legend()

        # 2. Target Successes Progress (NEW)
        ax2 = plt.subplot(3, 3, 2)
        cumulative_successes = np.cumsum(iteration_target_successes) if iteration_target_successes else np.array([])
        if cumulative_successes.size:
            plt.plot(iterations[:len(cumulative_successes)], cumulative_successes, marker='o', color='green',
                     linewidth=2, label='Cumulative')
        if iteration_target_successes:
            plt.bar(iterations[:len(iteration_target_successes)], iteration_target_successes, alpha=0.3,
                    color='lightgreen', label='Per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Target Successes')
        plt.title('Target Successes Over Iterations')
        plt.legend()

        # 3. Consecutive Failures Tracking (NEW)
        ax3 = plt.subplot(3, 3, 3)
        plt.step([0] + iterations[:len(iteration_consecutive_failures)],
                 [0] + iteration_consecutive_failures,
                 marker='o', color='red', linewidth=2, where='post')
        if result.stop_reasons:
            # Extract max consecutive failures from stop reasons if available
            for reason in result.stop_reasons:
                if 'consecutive failures' in reason.lower():
                    import re
                    match = re.search(r'\((\d+)\)', reason)
                    if match:
                        max_failures = int(match.group(1))
                        plt.axhline(y=max_failures, color='darkred', linestyle='--',
                                    label=f'Threshold ({max_failures})')
                        break
        plt.xlabel('Iteration')
        plt.ylabel('Consecutive Failures')
        plt.title('Consecutive Failures Tracking')
        plt.legend()

        # 4. Distribution of Suggested Labels (DISCRETE ONLY)
        if is_discrete:
            ax4 = plt.subplot(3, 3, 4)
            optimization_targets = self.al_campaign_config.discrete_targets

            # Extract labels from iteration results (use aligned iteration_results)
            iteration_labels = []
            id2label = {data_point.seq_id: data_point.label for data_point in self.al_simulation_config.simulation_data}
            for iter_result in iteration_results:
                iter_labels = [id2label[suggestion] for suggestion in iter_result.suggestions]
                iteration_labels.append(iter_labels)

            # Get unique labels
            unique_labels = sorted(
                list(set([data_point.label for data_point in self.al_simulation_config.simulation_data])))

            # Create cumulative counts for each label
            label_counts = []
            for iter_labels in iteration_labels:
                counts = {label: iter_labels.count(label) for label in unique_labels}
                label_counts.append(counts)

            # Create stacked bar chart (iterations length must match iteration_labels length)
            bar_iters = list(range(1, len(iteration_labels) + 1))
            bottom = np.zeros(len(iteration_labels), dtype=float)
            for label in unique_labels:
                values = np.array([counts[label] for counts in label_counts], dtype=float)
                label_display = f"★ {label}" if optimization_targets and label in optimization_targets else label
                plt.bar(bar_iters, values, bottom=bottom, label=label_display)
                bottom += values

            plt.xlabel('Iteration')
            plt.ylabel('Cumulative Suggested Labels')
            plt.title('Distribution of Suggested Labels')
            plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')

        # plt.savefig(f'{result.campaign_name}_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _print_stats(result: ActiveLearningSimulationResult):
        print(f"Simulation campaign stats:")
        print(f"Simulation stop reasons: {result.stop_reasons}")
        print(f"Total number of iterations: {len(result.iteration_results or [])}")
        print(f"Metrics for all masked data points per iteration: {result.iteration_metrics_total}")
        print(f"Metrics for suggested data points per iteration: {result.iteration_metrics_suggestions}")
        print(f"Target successes over iterations: {result.iteration_target_successes}")
        filtered_results_suggestions = [[sugg for sugg in res.results if sugg.entity_id in res.suggestions][0]
                                        for res in result.iteration_results or []]
        print(f"Iteration result for top suggestion: {filtered_results_suggestions}")

    def visualize(self):
        self._print_stats(self.simulation_result)
        self._visualize_results()

    def label(self) -> str:
        seed = self.al_campaign_config.seed
        model_type = getattr(self.al_campaign_config.model_type, "value",
                             self.al_campaign_config.model_type)
        return f"seed={seed} | {model_type} | success={self.is_success()}"

    def build_altair_charts(self):
        """Build interactive Altair charts mirroring `_visualize_results()`.

        Returns a dict keyed by chart name. The 'suggested_labels' key is
        only present for DISCRETE optimization mode.
        """
        import al_plots

        result = self.simulation_result
        is_discrete = self.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "RMSE"

        n_rounds_candidates = [
            len(getattr(result, "iteration_results", []) or []),
            len(getattr(result, "iteration_metrics_total", []) or []),
            len(getattr(result, "iteration_metrics_suggestions", []) or []),
            len(getattr(result, "iteration_target_successes", []) or []),
            len(getattr(result, "iteration_consecutive_failures", []) or []),
        ]
        n_rounds = max(n_rounds_candidates) if any(n_rounds_candidates) else 0

        iteration_metrics_total = [m.mean for m in (result.iteration_metrics_total or [])[:n_rounds]]
        iteration_metrics_suggestions = [m.mean for m in (result.iteration_metrics_suggestions or [])[:n_rounds]]
        iteration_target_successes = (result.iteration_target_successes or [])[:n_rounds]
        iteration_consecutive_failures = (result.iteration_consecutive_failures or [])[:n_rounds]
        iteration_results = (result.iteration_results or [])[:n_rounds]

        charts = {
            "metric_evolution": al_plots.build_metric_evolution_chart(
                iteration_metrics_total=list(iteration_metrics_total),
                iteration_metrics_suggestions=list(iteration_metrics_suggestions),
                iteration_target_successes=list(iteration_target_successes),
                metric_name=metric_name,
            ),
            "target_successes": al_plots.build_target_successes_chart(
                iteration_target_successes=list(iteration_target_successes),
            ),
            "consecutive_failures": al_plots.build_consecutive_failures_chart(
                iteration_consecutive_failures=list(iteration_consecutive_failures),
                stop_reasons=list(result.stop_reasons or []),
            ),
        }

        if is_discrete:
            id2label = {dp.seq_id: dp.label for dp in self.al_simulation_config.simulation_data}
            iteration_suggestions = [list(ir.suggestions) for ir in iteration_results]
            unique_labels = sorted({dp.label for dp in self.al_simulation_config.simulation_data})
            charts["suggested_labels"] = al_plots.build_suggested_labels_chart(
                iteration_suggestions=iteration_suggestions,
                id2label=id2label,
                unique_labels=unique_labels,
                optimization_targets=self.al_campaign_config.discrete_targets,
            )

        return charts


class ActiveLearningMultipleSimulationResult:
    def __init__(self, simulation_results: List[ActiveLearningSingleSimulationResult]):
        self.simulation_results = simulation_results
        self._validate()

    def _validate(self):
        results = self.simulation_results
        first_result = results[0]
        for result in results:
            assert first_result.al_campaign_config.embedder_name == result.al_campaign_config.embedder_name, "Embedder config must be the same"
            assert first_result.al_campaign_config.optimization_mode == result.al_campaign_config.optimization_mode, "Optimization mode must be the same"
            assert first_result.al_simulation_config.convergence_config.target_successes == result.al_simulation_config.convergence_config.target_successes, "Simulation configs must be the same"

    @classmethod
    def from_json(cls, path: Path) -> ActiveLearningMultipleSimulationResult:
        """Load simulation results from JSON file"""
        with open(path, 'r') as f:
            json_results = json.load(f)
            # Convert dict to JSON string first, then validate
            results = [
                ActiveLearningSingleSimulationResult.model_validate_json(json.dumps(res))
                for res in json_results
            ]
            return cls(results)

    def get_best_simulation(self):
        return max(self.simulation_results, key=lambda ssr: ssr.simulation_result.iteration_target_successes)

    def get_worst_simulation(self):
        return min(self.simulation_results, key=lambda ssr: ssr.simulation_result.iteration_target_successes)

    def _percent_successful(self):
        return sum([1 for ssr in self.simulation_results if ssr.is_success()]) / len(self.simulation_results) * 100

    def _print_stats(self):
        print(f"Summary over {len(self.simulation_results)} simulations:")
        print(f"Percent successful: {self._percent_successful()}%")

    def _visualize(self):
        """Visualize aggregated results across multiple simulations - Top 3 most informative plots"""

        is_discrete = self.simulation_results[
                          0].al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "RMSE"

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(18, 6))

        # Prepare common data
        max_iterations = max([len(ssr.simulation_result.iteration_metrics_total or [])
                              for ssr in self.simulation_results])
        iterations = np.arange(1, max_iterations + 1)

        # ========================================================================
        # PLOT 1: Success Rate & Convergence Speed Overview
        # ========================================================================
        ax1 = plt.subplot(1, 3, 1)

        # Top: Success rate as horizontal bar
        success_count = sum([1 for ssr in self.simulation_results if ssr.is_success()])
        failure_count = len(self.simulation_results) - success_count
        success_rate = self._percent_successful()

        # Create a split bar chart showing success metrics
        y_pos = [0, 1, 2]

        # Success rate
        ax1.barh(y_pos[0], success_rate, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax1.barh(y_pos[0], 100 - success_rate, left=success_rate, color='#e74c3c', alpha=0.7, edgecolor='black')

        # Calculate convergence stats
        convergence_iterations = []
        for ssr in self.simulation_results:
            cumsum = np.cumsum(ssr.simulation_result.iteration_target_successes)
            target = ssr.al_simulation_config.convergence_config.target_successes
            converged_at = np.where(cumsum >= target)[0]
            if len(converged_at) > 0:
                convergence_iterations.append(converged_at[0] + 1)

        if convergence_iterations:
            avg_convergence = np.mean(convergence_iterations)
            std_convergence = np.std(convergence_iterations)

            # Normalize convergence speed to 0-100 scale (inverse: faster = higher score)
            # Assuming max reasonable iterations is max_iterations
            convergence_score = max(0, 100 * (1 - avg_convergence / max_iterations))

            ax1.barh(y_pos[1], convergence_score, color='#3498db', alpha=0.7, edgecolor='black')
            ax1.barh(y_pos[1], 100 - convergence_score, left=convergence_score, color='#bdc3c7', alpha=0.7,
                     edgecolor='black')

            # Add text annotations
            ax1.text(success_rate / 2, y_pos[0], f'{success_rate:.1f}%',
                     ha='center', va='center', fontweight='bold', fontsize=11)
            ax1.text(success_rate + (100 - success_rate) / 2, y_pos[0], f'{100 - success_rate:.1f}%',
                     ha='center', va='center', fontweight='bold', fontsize=11)
            ax1.text(50, y_pos[1], f'Avg: {avg_convergence:.1f}±{std_convergence:.1f} iter',
                     ha='center', va='center', fontweight='bold', fontsize=10)

            convergence_label = f'Convergence Speed\n(of {len(convergence_iterations)} successful)'
        else:
            convergence_label = 'Convergence Speed\n(none converged)'

        ax1.set_yticks(y_pos[:2])
        ax1.set_yticklabels([f'Success Rate\n({success_count}/{len(self.simulation_results)} runs)',
                             convergence_label])
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Percentage / Score')
        ax1.set_title('Overall Performance Summary', fontweight='bold', fontsize=12)
        ax1.legend(['Success/Fast', 'Failure/Slow'], loc='lower right')

        # ========================================================================
        # PLOT 2: Average Cumulative Target Successes Over Time
        # ========================================================================
        ax2 = plt.subplot(1, 3, 2)

        # Pad data
        target_successes_padded = []
        for ssr in self.simulation_results:
            successes = ssr.simulation_result.iteration_target_successes
            target_successes_padded.append(successes + [0] * (max_iterations - len(successes)))

        cumulative_successes = np.cumsum(np.array(target_successes_padded), axis=1)
        mean_cumulative = np.mean(cumulative_successes, axis=0)
        std_cumulative = np.std(cumulative_successes, axis=0)

        # Plot individual runs in background (light)
        for i, sim_cumulative in enumerate(cumulative_successes):
            color = '#2ecc71' if self.simulation_results[i].is_success() else '#e74c3c'
            ax2.plot(iterations, sim_cumulative, color=color, alpha=0.15, linewidth=1)

        # Plot mean with confidence band
        ax2.plot(iterations, mean_cumulative, marker='o', linewidth=3,
                 color='#16a085', label='Mean', zorder=10)
        ax2.fill_between(iterations,
                         np.maximum(0, mean_cumulative - std_cumulative),
                         mean_cumulative + std_cumulative,
                         alpha=0.3, color='#16a085', label='±1 SD')

        # Target line
        target = self.simulation_results[0].al_simulation_config.convergence_config.target_successes
        ax2.axhline(y=target, color='red', linestyle='--', linewidth=2,
                    label=f'Target ({target})', zorder=5)

        ax2.set_xlabel('Iteration', fontweight='bold')
        ax2.set_ylabel('Cumulative Target Successes', fontweight='bold')
        ax2.set_title('Target Discovery Progress', fontweight='bold', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # ========================================================================
        # PLOT 3: Average Model Performance Evolution
        # ========================================================================
        ax3 = plt.subplot(1, 3, 3)

        # Pad metrics data
        metrics_total_padded = []
        for ssr in self.simulation_results:
            metrics_total = [m.mean for m in ssr.simulation_result.iteration_metrics_total]
            metrics_total_padded.append(metrics_total + [np.nan] * (max_iterations - len(metrics_total)))

        metrics_total_array = np.array(metrics_total_padded)
        mean_total = np.nanmean(metrics_total_array, axis=0)
        std_total = np.nanstd(metrics_total_array, axis=0)

        # Get best and worst for comparison
        best_result = self.get_best_simulation()
        worst_result = self.get_worst_simulation()

        best_iters = range(1, len(best_result.simulation_result.iteration_metrics_total) + 1)
        worst_iters = range(1, len(worst_result.simulation_result.iteration_metrics_total) + 1)

        # Plot best and worst in background
        ax3.plot(worst_iters, [m.mean for m in worst_result.simulation_result.iteration_metrics_total],
                 marker='v', label='Worst Run', linewidth=2, color='#e74c3c', alpha=0.6, linestyle='--')
        ax3.plot(best_iters, [m.mean for m in best_result.simulation_result.iteration_metrics_total],
                 marker='^', label='Best Run', linewidth=2, color='#2ecc71', alpha=0.6, linestyle='--')

        # Plot mean with confidence band (emphasize this)
        ax3.plot(iterations, mean_total, marker='o', label='Mean',
                 linewidth=3, color='#3498db', zorder=10)
        ax3.fill_between(iterations, mean_total - std_total, mean_total + std_total,
                         alpha=0.3, color='#3498db', label='±1 SD')

        ax3.set_xlabel('Iteration', fontweight='bold')
        ax3.set_ylabel(f'{metric_name}', fontweight='bold')
        ax3.set_title(f'Model Performance Evolution ({metric_name})', fontweight='bold', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def visualize(self):
        self._print_stats()
        print("Visualizing aggregated results across multiple simulations...")
        self._visualize()
        print("Visualizating worst simulation...")
        self.get_worst_simulation().visualize()
        print("Visualizating best simulation...")
        self.get_best_simulation().visualize()

    def save(self, path: Path):
        """Save simulation results to JSON file"""
        with open(path, 'w') as f:
            json_results = []
            for result in self.simulation_results:
                # Parse the result into dict structure
                result_dict = json.loads(result.model_dump_json())

                # Remove simulation_data from al_simulation_config to reduce file size
                if 'al_simulation_config' in result_dict and 'simulation_data' in result_dict['al_simulation_config']:
                    result_dict['al_simulation_config']['simulation_data'] = [
                        SequenceData(seq_id="Dummy1", seq="MDUMMY").model_dump(),
                        SequenceData(seq_id="Dummy2", seq="ADUMMY").model_dump(),
                        SequenceData(seq_id="Dummy3", seq="GDUMMY").model_dump()
                    ]

                json_results.append(result_dict)

            # Write the JSON with proper formatting
            json.dump(json_results, f, indent=4)

    def build_altair_charts(self):
        """Build interactive Altair charts mirroring `_visualize()`.

        Returns a dict with keys: 'performance_summary',
        'mean_cumulative_successes', 'mean_metric_evolution'.
        """
        import al_plots

        is_discrete = self.simulation_results[
                          0].al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "RMSE"

        per_sim_target_successes = [
            list(ssr.simulation_result.iteration_target_successes or [])
            for ssr in self.simulation_results
        ]
        per_sim_metrics_total = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_total or []]
            for ssr in self.simulation_results
        ]
        per_sim_is_success = [ssr.is_success() for ssr in self.simulation_results]

        target_threshold = self.simulation_results[0].al_simulation_config.convergence_config.target_successes
        max_iterations = max((len(m) for m in per_sim_metrics_total), default=0)
        success_count = sum(1 for s in per_sim_is_success if s)

        convergence_iterations, _, _ = al_plots._compute_convergence_stats(
            per_sim_target_successes=per_sim_target_successes,
            target_successes_threshold=target_threshold,
        )

        best = self.get_best_simulation()
        worst = self.get_worst_simulation()

        return {
            "performance_summary": al_plots.build_performance_summary_chart(
                success_count=success_count,
                total_runs=len(self.simulation_results),
                convergence_iterations=convergence_iterations,
                max_iterations=max_iterations,
            ),
            "mean_cumulative_successes": al_plots.build_mean_cumulative_successes_chart(
                per_sim_target_successes=per_sim_target_successes,
                per_sim_is_success=per_sim_is_success,
                target_threshold=target_threshold,
            ),
            "mean_metric_evolution": al_plots.build_mean_metric_evolution_chart(
                per_sim_metrics_total=per_sim_metrics_total,
                best_run_metrics=[m.mean for m in best.simulation_result.iteration_metrics_total or []],
                worst_run_metrics=[m.mean for m in worst.simulation_result.iteration_metrics_total or []],
                metric_name=metric_name,
            ),
        }

    def summary(self) -> dict:
        """Return a small dict of summary values for the dashboard header."""
        first = self.simulation_results[0]
        return {
            "embedder_name": first.al_campaign_config.embedder_name,
            "model_type": getattr(first.al_campaign_config.model_type, "value",
                                  first.al_campaign_config.model_type),
            "optimization_mode": getattr(first.al_campaign_config.optimization_mode, "value",
                                         first.al_campaign_config.optimization_mode),
            "n_simulations": len(self.simulation_results),
            "n_successful": sum(1 for ssr in self.simulation_results if ssr.is_success()),
            "target_successes_threshold": first.al_simulation_config.convergence_config.target_successes,
            "is_discrete": first.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE,
            "discrete_targets": first.al_campaign_config.discrete_targets,
        }


class ActiveLearningSimulationComparer:
    pass  # TODO
