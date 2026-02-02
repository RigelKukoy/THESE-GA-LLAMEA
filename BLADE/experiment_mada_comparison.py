"""
MADA-LLAMEA Comparison Experiment

Compares MADA-LLAMEA (with Discounted Thompson Sampling) against baseline methods:
- LLaMEA: Standard evolutionary algorithm
- EoH: Evolution of Heuristics (optional)

This demonstrates how adaptive operator selection improves performance and
exploration-exploitation balance compared to static operator probabilities.
"""
import os
from iohblade.llm import OpenAI_LLM
from iohblade.methods import LLaMEA, MADA_LLaMEA
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.plots import (
    plot_convergence,
    plot_boxplot_fitness_hue,
    plot_experiment_CEG,
)
import openai


class AIML_LLM(OpenAI_LLM):
    """AIML API LLM (OpenAI-compatible)"""

    def __init__(
        self,
        api_key,
        model="gemini-2.0-flash",
        base_url="https://api.aimlapi.com/v1",
        temperature=0.8,
        **kwargs
    ):
        super(OpenAI_LLM, self).__init__(api_key, model, base_url, **kwargs)
        self._client_kwargs = dict(api_key=api_key, base_url=base_url)
        self.client = openai.OpenAI(**self._client_kwargs)
        self.temperature = temperature


def create_methods(llm, budget):
    """Create methods to compare."""
    methods = [
        # Baseline LLaMEA
        LLaMEA(
            llm=llm,
            budget=budget,
            name="LLaMEA-Baseline",
            n_parents=4,
            n_offspring=8,
            elitism=True,
        ),
        # MADA-LLAMEA with default discount (0.9)
        MADA_LLaMEA(
            llm=llm,
            budget=budget,
            name="MADA-LLaMEA-0.9",
            n_parents=4,
            n_offspring=8,
            elitism=True,
            discount=0.9,  # Moderate adaptation
            tau_max=1.0,
            reward_variance=1.0,
        ),
        # MADA-LLAMEA with fast adaptation (0.8)
        MADA_LLaMEA(
            llm=llm,
            budget=budget,
            name="MADA-LLaMEA-0.8",
            n_parents=4,
            n_offspring=16,
            elitism=True,
            discount=0.8,  # Faster adaptation
            tau_max=1.0,
            reward_variance=1.0,
        ),
        # MADA-LLAMEA with slow adaptation (0.95)
        MADA_LLaMEA(
            llm=llm,
            budget=budget,
            name="MADA-LLaMEA-0.95",
            n_parents=4,
            n_offspring=16,
            elitism=True,
            discount=0.95,  # Slower adaptation
            tau_max=1.0,
            reward_variance=1.0,
        ),
    ]

    # Optionally add EoH if available
    try:
        from iohblade.methods import EoH

        methods.append(
            EoH(llm=llm, budget=budget, name="EoH-Baseline", pop_size=20)
        )
    except ImportError:
        print("⚠️  EoH not available (requires additional dependencies)")

    return methods


def visualize_operator_selection(logger, experiment_name):
    """Create MADA-specific visualizations for operator selection."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    print("\n📊 Generating MADA-specific visualizations...")

    try:
        # Get data
        df = logger.get_problem_data()

        # Filter MADA methods only
        mada_methods = [
            name for name in df["method_name"].unique() if "MADA" in name
        ]

        if not mada_methods:
            print("   No MADA methods found in results")
            return

        df_mada = df[df["method_name"].isin(mada_methods)].copy()

        # Extract operator from metadata
        df_mada["operator"] = df_mada["metadata"].apply(
            lambda x: x.get("operator", "unknown") if isinstance(x, dict) else "unknown"
        )
        df_mada["generation"] = df_mada["metadata"].apply(
            lambda x: x.get("generation", 0) if isinstance(x, dict) else 0
        )
        df_mada["reward"] = df_mada["metadata"].apply(
            lambda x: x.get("reward", 0) if isinstance(x, dict) else 0
        )

        # 1. Operator selection over generations (stacked area)
        for method in mada_methods:
            df_method = df_mada[df_mada["method_name"] == method]

            op_counts = (
                df_method.groupby(["generation", "operator"])
                .size()
                .unstack(fill_value=0)
            )

            if op_counts.empty:
                continue

            plt.figure(figsize=(12, 6))
            op_counts.plot(kind="area", stacked=True, alpha=0.7, ax=plt.gca())
            plt.title(f"{method}: Operator Selection Over Generations")
            plt.xlabel("Generation")
            plt.ylabel("Number of Selections")
            plt.legend(title="Operator", loc="upper left")
            plt.tight_layout()
            filename = f"{logger.dirname}/{method}_operator_selection.png"
            plt.savefig(filename)
            plt.close()
            print(f"   ✓ Saved: {filename}")

        # 2. Reward distribution by operator (all MADA methods)
        plt.figure(figsize=(12, 6))
        df_valid = df_mada[df_mada["operator"] != "unknown"]
        if not df_valid.empty:
            sns.boxplot(data=df_valid, x="operator", y="reward", hue="method_name")
            plt.title("Reward Distribution by Operator (All MADA Methods)")
            plt.ylabel("Fitness Improvement")
            plt.xlabel("Operator")
            plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            filename = f"{logger.dirname}/mada_reward_distribution.png"
            plt.savefig(filename)
            plt.close()
            print(f"   ✓ Saved: {filename}")

        # 3. Operator selection percentages (summary table)
        print("\n📈 Operator Selection Summary:")
        for method in mada_methods:
            df_method = df_mada[df_mada["method_name"] == method]
            op_pct = (
                df_method["operator"].value_counts(normalize=True) * 100
            ).round(1)
            print(f"\n   {method}:")
            for op, pct in op_pct.items():
                if op != "unknown":
                    print(f"      {op:15s}: {pct:5.1f}%")

    except Exception as e:
        print(f"   ⚠️  Error generating MADA visualizations: {e}")


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════════

    # Get credentials
    api_key = os.getenv("AIML_API_KEY")
    base_url = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")

    if not api_key:
        print("❌ AIML_API_KEY environment variable not set")
        print("\nPlease set your API key:")
        print("  export AIML_API_KEY='your-api-key-here'")
        exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # Experiment Parameters
    # ═══════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("MADA-LLAMEA Comparison Experiment")
    print("=" * 80)
    print("\nExperimental Setup:")
    print("  Comparing MADA-LLAMEA with adaptive operator selection against baselines")
    print("=" * 80)

    # Paper specification (Section 4.1.1)
    budget = 100  # Generate 100 candidate algorithms
    RUNS = 1      # Multiple runs for statistical significance
    dims = [5]    # 5-dimensional as per paper
    budget_factor = 2000  # 2000 × dim = 10,000 function evaluations
    
    # 20 Train / 50 Test split as specified
    training_instances = list(range(20))  # Instances 0-19 (20 total)
    test_instances = list(range(20, 70))  # Instances 20-69 (50 total)

    print("\nConfiguration:")
    print(f"  Budget: {budget} candidate algorithms per run")
    print(f"  Runs: {RUNS} independent runs")
    print(f"  Dimension: {dims[0]}D")
    print(f"  Function Evaluations: {budget_factor} × {dims[0]} = {budget_factor * dims[0]:,} per algorithm")
    print(f"  Training Instances: {len(training_instances)} instances ({min(training_instances)}-{max(training_instances)})")
    print(f"  Test Instances: {len(test_instances)} instances ({min(test_instances)}-{max(test_instances)})")
    print(f"  Total Training LLM Calls: ~{budget * RUNS * len(training_instances):,}")
    print(f"  Total Test LLM Calls: ~{budget * RUNS * len(test_instances):,}")

    print("\nMethods:")
    print("  1. LLaMEA-Baseline:    Standard evolutionary algorithm")
    print("  2. MADA-LLaMEA-0.9:    D-TS with discount=0.9 (moderate adaptation)")
    print("  3. MADA-LLaMEA-0.8:    D-TS with discount=0.8 (fast adaptation)")
    print("  4. MADA-LLaMEA-0.95:   D-TS with discount=0.95 (slow adaptation)")
    print("  5. EoH-Baseline:       Evolution of Heuristics (if available)")

    print("\nExpected Outcomes:")
    print("  - MADA should adaptively select best operators over time")
    print("  - Different discount factors adapt at different speeds")
    print("  - Operator selection should vary by problem characteristics")
    print("  - Overall performance should match or exceed baselines")
    print("  - Training performance vs Test generalization will be compared")

    # Estimate time
    total_instances = len(training_instances) + len(test_instances)
    estimated_mins = (budget * RUNS * total_instances) // 10
    print(f"\n⏱️  Estimated time: {estimated_mins}-{estimated_mins*2} minutes")
    print()

    # User confirmation
    user_confirm = input(
        "This will consume API credits. Continue? [y/N]: "
    ).strip().lower()
    if user_confirm != "y":
        print("Experiment cancelled.")
        exit(0)

    # ═══════════════════════════════════════════════════════════════════
    # Initialize LLM
    # ═══════════════════════════════════════════════════════════════════

    print("\n🔧 Initializing LLM...")
    llm = AIML_LLM(
        api_key=api_key,
        model="gemini-2.0-flash",
        base_url=base_url,
        temperature=0.8,
    )
    print("   ✓ LLM initialized: gemini-2.0-flash")

    # ═══════════════════════════════════════════════════════════════════
    # Create Methods
    # ═══════════════════════════════════════════════════════════════════

    print("\n🔧 Creating methods...")
    methods = create_methods(llm, budget)
    print(f"   ✓ {len(methods)} methods created")
    for method in methods:
        print(f"      - {method.name}")

    # ═══════════════════════════════════════════════════════════════════
    # Create Problems
    # ═══════════════════════════════════════════════════════════════════

    print("\n🔧 Creating problem with train/test split...")
    problem = MA_BBOB(
        training_instances=training_instances,
        test_instances=test_instances,
        dims=dims,
        budget_factor=budget_factor,
        name='MA-BBOB-MADA-Comparison',
        eval_timeout=600  # 10 minutes timeout per evaluation
    )
    problems = [problem]
    print(f"   ✓ Problem created with {len(training_instances)} training + {len(test_instances)} test instances")

    # ═══════════════════════════════════════════════════════════════════
    # Setup Logger
    # ═══════════════════════════════════════════════════════════════════

    experiment_name = "MADA_Comparison"
    logger = ExperimentLogger(f"results/{experiment_name}")
    print(f"\n📁 Results directory: {logger.dirname}")

    # ═══════════════════════════════════════════════════════════════════
    # Run Experiment
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("Starting Experiment...")
    print("=" * 80)
    print()

    try:
        experiment = Experiment(
            methods=methods,
            problems=problems,
            runs=RUNS,
            show_stdout=True,
            exp_logger=logger,
        )

        experiment()

        print()
        print("=" * 80)
        print("✅ Experiment completed successfully!")
        print("=" * 80)

        # ═══════════════════════════════════════════════════════════════
        # Generate Visualizations
        # ═══════════════════════════════════════════════════════════════

        print("\n📊 Generating standard visualizations...")

        try:
            # Convergence plot
            print("   - Convergence plot...")
            plot_convergence(logger, metric="AOCC", save=True, budget=budget)
            print(f"     ✓ Saved: {logger.dirname}/convergence.png")

            # Boxplot with method comparison
            print("   - Fitness boxplot...")
            plot_boxplot_fitness_hue(logger, hue="method_name", save=True)
            print(f"     ✓ Saved: {logger.dirname}/boxplot_fitness.png")

            # CEG plot
            print("   - Cumulative Expected Gain plot...")
            plot_experiment_CEG(logger, save=True, budget=budget, max_seeds=RUNS)
            print(f"     ✓ Saved: {logger.dirname}/ceg.png")

        except Exception as e:
            print(f"   ⚠️  Warning: Some visualizations failed: {e}")

        # Generate MADA-specific visualizations
        visualize_operator_selection(logger, experiment_name)

        # ═══════════════════════════════════════════════════════════════
        # Summary Statistics
        # ═══════════════════════════════════════════════════════════════

        print("\n" + "=" * 80)
        print("📈 Summary Statistics")
        print("=" * 80)

        try:
            df = logger.get_problem_data()

            print("\nFinal Fitness (mean ± std):")
            summary = (
                df.groupby("method_name")["fitness"]
                .agg(["mean", "std", "count"])
                .round(4)
            )
            for method, row in summary.iterrows():
                print(
                    f"  {method:25s}: {row['mean']:8.4f} ± {row['std']:8.4f} (n={int(row['count'])})"
                )

            # Best method
            best_method = summary["mean"].idxmax()
            print(f"\n🏆 Best performing method: {best_method}")

        except Exception as e:
            print(f"   ⚠️  Could not compute summary statistics: {e}")

        # ═══════════════════════════════════════════════════════════════
        # Next Steps
        # ═══════════════════════════════════════════════════════════════

        print("\n" + "=" * 80)
        print("Next Steps")
        print("=" * 80)
        print(f"\n1. Check detailed results in: {logger.dirname}")
        print("2. Review experimentlog.jsonl for per-evaluation data")
        print("3. Analyze operator selection plots for MADA methods")
        print("4. Compare MADA variants (discount 0.8, 0.9, 0.95)")
        print("5. Run statistical significance tests")
        print("\nKey Questions to Answer:")
        print("  - Does MADA outperform baseline LLaMEA?")
        print("  - Which discount factor works best?")
        print("  - How does operator selection adapt over time?")
        print("  - Training vs Test performance: Does MADA generalize better?")
        print("  - Are there problem-specific operator preferences?")
        print("\nKey metrics to analyze:")
        print("  - Training performance: Performance on 20 training instances")
        print("  - Test performance: Generalization to 50 unseen test instances")
        print("  - Overfitting: Compare training vs test performance gap")
        print("=" * 80)

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("⚠️  Experiment interrupted by user")
        print("=" * 80)
        print(f"Partial results may be available in: {logger.dirname}")
        print("=" * 80)
        raise

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Experiment failed!")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nCommon issues:")
        print("  - Check AIML API credentials are valid")
        print("  - Verify sufficient API credits/quota")
        print("  - Check network connectivity")
        print("  - Ensure sufficient disk space for results")
        print("  - Check eval_timeout if evaluations are timing out")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        raise

