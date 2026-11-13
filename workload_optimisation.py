"""Run all coursework experiments with: python workload_optimisation.py

Place both supplied instance text files beside this script.
Dependencies: NumPy, SciPy and Matplotlib.
"""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
POPULATION = 80
BUDGET = 20_000


# Supplied dataset loader

def load_workload_instance(file_path):
    """
    Loads a hardware-constrained workload allocation dataset.
    
    Parameters:
        file_path (str): Path to the dataset plain text file (.txt).
        
    Returns:
        n (int): Number of candidate AI models.
        m (int): Number of hardware resource constraints.
        utilities (np.ndarray): 1D array of shape (n,) containing model utility scores (p_j).
        demands (np.ndarray): 2D array of shape (m, n) containing resource consumption demands (r_{i,j}).
        capacities (np.ndarray): 1D array of shape (m,) containing hardware capacity limits (b_i).
    """
    with open(file_path, 'r') as f:
        tokens = f.read().split()
        
    if not tokens:
        raise ValueError(f"Dataset file at '{file_path}' is empty or invalid.")
        
    n = int(tokens[0])
    m = int(tokens[1])
    idx = 2
    
    # Model utility scores (length n)
    utilities = np.array([float(x) for x in tokens[idx : idx + n]])
    idx += n
    
    # Resource demand matrix (shape m x n)
    demands = np.zeros((m, n))
    for i in range(m):
        demands[i, :] = [float(x) for x in tokens[idx : idx + n]]
        idx += n
        
    # Maximum hardware capacity limits (length m)
    capacities = np.array([float(x) for x in tokens[idx : idx + m]])
    
    return n, m, utilities, demands, capacities


# Part 1: hill-climber and baseline evaluation

def read_instance(name):
    path = ROOT / name
    model_count, resource_count, utilities, demands, capacities = (
        load_workload_instance(path)
    )
    assert (
        len(path.read_text().split())
        == 2 + model_count + resource_count * model_count + resource_count
    )
    assert demands.shape == (resource_count, model_count) and capacities.shape == (
        resource_count,
    )
    assert all(
        np.all(values == values.astype(int))
        for values in (utilities, demands, capacities)
    )
    assert np.all(demands >= 0) and np.all(capacities >= 0)
    return utilities.astype(int), demands.astype(int), capacities.astype(int)

def hill_climb(utilities, demands, capacities, seed):
    rng = np.random.default_rng(seed)
    selection = np.zeros(len(utilities), dtype=bool)
    usage = np.zeros_like(capacities)
    for model_index in rng.permutation(len(utilities)):
        if np.all(usage + demands[:, model_index] <= capacities):
            selection[model_index] = True
            usage += demands[:, model_index]
    current_utility = int(utilities @ selection)
    # Count the completed start once, then every feasible or infeasible neighbour.
    evaluations = 1
    history = [(evaluations, current_utility)]
    while True:
        best_utility = current_utility
        best_move = None
        selected = np.flatnonzero(selection).tolist()
        available = np.flatnonzero(~selection)
        # Batch each drop's possible additions in the original scan order.
        addition_demands = demands[:, available]
        addition_values = utilities[available]
        for removed_model in [None] + selected:
            base_usage = (
                usage if removed_model is None else usage - demands[:, removed_model]
            )
            base_value = (
                current_utility
                if removed_model is None
                else current_utility - utilities[removed_model]
            )
            evaluations += len(available)
            values = base_value + addition_values
            feasible = np.all(
                base_usage[:, None] + addition_demands <= capacities[:, None], axis=0
            )
            improving = np.flatnonzero(feasible & (values > best_utility))
            if improving.size:
                # argmax keeps the first tie, matching the original scalar loop.
                position = improving[np.argmax(values[improving])]
                best_utility = int(values[position])
                best_move = (removed_model, available[position])
        if best_move is None:
            history.append((evaluations, current_utility))
            return selection, history
        removed_model, added_model = best_move
        if removed_model is not None:
            selection[removed_model] = False
            usage -= demands[:, removed_model]
        selection[added_model] = True
        usage += demands[:, added_model]
        current_utility = best_utility
        history.append((evaluations, current_utility))

def exact_reference(utilities, demands, capacities):
    # Two-capacity 0-1 dynamic programming. Read only the previous item layer.
    table = np.zeros(tuple(capacities + 1), dtype=int)
    for profit, (resource_1_cost, resource_2_cost) in zip(utilities, demands.T):
        if resource_1_cost <= capacities[0] and resource_2_cost <= capacities[1]:
            previous = table[
                : capacities[0] - resource_1_cost + 1,
                : capacities[1] - resource_2_cost + 1,
            ]
            target = table[resource_1_cost:, resource_2_cost:]
            np.maximum(target, previous + profit, out=target)
    return int(table[-1, -1])

def run_baseline():
    utilities, demands, capacities = read_instance("multi_resource_n28_m2.txt")
    large_p, large_r, large_b = read_instance("multi_resource_n90_m5.txt")
    print("Challenge input:", len(large_p), large_r.shape, large_b)
    optimum = exact_reference(utilities, demands, capacities)
    rows, curves = [], []
    for seed in range(30):
        selection, trace = hill_climb(utilities, demands, capacities, seed)
        selected_models = np.flatnonzero(selection)
        resource_usage = demands[:, selected_models].sum(axis=1)
        current_utility = int(utilities[selected_models].sum())
        assert (
            current_utility == trace[-1][1]
            and current_utility <= optimum
            and np.all(resource_usage <= capacities)
        )
        # Independent local optimality check using complete candidate bit vectors.
        for added_model in np.flatnonzero(~selection):
            for removed_model in [None] + selected_models.tolist():
                trial = selection.copy()
                trial[added_model] = True
                if removed_model is not None:
                    trial[removed_model] = False
                assert utilities @ trial <= current_utility or np.any(
                    demands @ trial > capacities
                )
        rows.append(
            [
                seed,
                trace[0][1],
                current_utility,
                trace[-1][0],
                *resource_usage,
                " ".join(map(str, selected_models + 1)),
            ]
        )
        curves.append(np.array(trace))
    with (ROOT / "runs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Random Seed",
                "Starting Utility Score",
                "Final Utility Score",
                "Solutions Evaluated",
                "First Capacity Used (Limit 500)",
                "Second Capacity Used (Limit 500)",
                "Selected Model Numbers",
            ]
        )
        writer.writerows(rows)
    grid = np.arange(1, max(t[-1, 0] for t in curves) + 1)
    values = [t[np.searchsorted(t[:, 0], grid, side="right") - 1, 1] for t in curves]
    low, median, high = np.percentile(values, [25, 50, 75], axis=0)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.fill_between(grid, low, high, step="post", alpha=0.22, label="Middle 50%")
    axes.step(grid, median, where="post", label="Median accepted utility")
    axes.axhline(optimum, ls="--", color="black", label="Exact optimum")
    axes.set(
        xlabel="Candidate evaluations",
        ylabel="Utility",
        title="Hill-climber on 28 models · 30 starts",
    )
    axes.legend()
    figure.savefig(ROOT / "convergence.png", dpi=160)
    scores = [row[2] for row in rows]
    print("Exact optimum, hits:", optimum, scores.count(optimum))
    print("Min, median, max:", min(scores), np.median(scores), max(scores))
    print("Median evaluations:", np.median([row[3] for row in rows]))
    print("Best run:", max(rows, key=lambda row: row[2]))


# Part 2: genetic algorithm and constraint handling

def initial_population(random_generator, demands, capacities):
    population = np.zeros((POPULATION, demands.shape[1]), dtype=bool)
    # Draw permutations in the same order, then pack all portfolios together.
    orders = np.array(
        [random_generator.permutation(demands.shape[1]) for _ in population]
    )
    remaining = np.broadcast_to(capacities, (POPULATION, len(capacities))).copy()
    rows = np.arange(POPULATION)
    for position in range(demands.shape[1]):
        models = orders[:, position]
        costs = demands[:, models].T
        fits = np.all(costs <= remaining, axis=1)
        population[rows[fits], models[fits]] = True
        remaining[fits] -= costs[fits]
    return population

def offspring(random_generator, population, fitness):
    """Two size-three tournaments per child, uniform crossover, bit mutation."""
    population_size, model_count = population.shape
    draws = random_generator.integers(population_size, size=(population_size, 2, 3))
    tournament_scores = fitness[draws]
    winning_positions = tournament_scores.argmax(axis=2)
    winners = np.take_along_axis(draws, winning_positions[..., None], axis=2)
    parents = population[winners[..., 0]]
    mask = random_generator.random((population_size, model_count)) < 0.5
    children = np.where(mask, parents[:, 0], parents[:, 1])
    mutation_mask = random_generator.random(children.shape) < 1 / model_count
    children ^= mutation_mask
    return children

def repair(children, usage, profit, demands, capacities):
    """Remove the item giving greatest normalised excess relief per utility lost."""
    removal_count = 0
    while True:
        excess = np.maximum(usage - capacities, 0)
        rows = np.flatnonzero(excess.any(axis=1))
        if not rows.size:
            return removal_count
        relief = np.zeros((len(rows), len(profit)))
        for resource_index, capacity in enumerate(capacities):
            relief += (
                np.minimum(excess[rows, resource_index, None], demands[resource_index])
                / capacity
            )
        scores = np.where(children[rows], relief / profit, -1)
        removed_models = scores.argmax(axis=1)
        children[rows, removed_models] = False
        usage[rows] -= demands[:, removed_models].T
        removal_count += len(rows)

def solve(initial, random_generator, utilities, demands, capacities, variant):
    if variant not in ("repair", "ranking"):
        raise ValueError(variant)
    population = initial.copy()
    fitness = population @ utilities
    best_selection = population[fitness.argmax()].copy()
    best_utility = int(fitness.max())
    history = [(len(initial), best_utility)]
    infeasible_count = removal_count = 0
    for evaluation_count in range(2 * POPULATION, BUDGET + 1, POPULATION):
        children = offspring(random_generator, population, fitness)
        usage = children @ demands.T
        infeasible_count += int(np.any(usage > capacities, axis=1).sum())
        if variant == "repair":
            removal_count += repair(children, usage, utilities, demands, capacities)
        values = children @ utilities
        fitness = values
        if variant == "ranking":
            violation = (np.maximum(usage - capacities, 0) / capacities).sum(axis=1)
            # Every feasible positive utility outranks a negative violation.
            fitness = np.where(violation > 0, -violation, values)
        leader = int(fitness.argmax())
        if fitness[leader] > best_utility:
            best_selection = children[leader].copy()
            best_utility = int(values[leader])
        population = children
        history.append((evaluation_count, best_utility))
    assert history[-1][0] == BUDGET
    assert (
        np.all(demands[:, best_selection].sum(axis=1) <= capacities)
        and int(utilities[best_selection].sum()) == best_utility
    )
    return best_selection, history, infeasible_count, removal_count


# Part 3: paired experiments

def run_experiments():
    _, _, utilities, demands, capacities = load_workload_instance(
        ROOT / "multi_resource_n90_m5.txt"
    )
    assert np.all(utilities > 0) and np.all(capacities > 0) and np.all(demands >= 0)
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "challenge_runs.csv").open("w", newline="") as runs_file, (
        RESULTS / "challenge_trace.csv"
    ).open("w", newline="") as trace_file:
        runs_writer, trace_writer = csv.writer(runs_file), csv.writer(trace_file)
        runs_writer.writerow(
            [
                "seed",
                "variant",
                "evaluations",
                "best",
                "search_seconds",
                "invalid_children",
                "offspring",
                "removed",
                "models_1_based",
                *[f"usage_{i}" for i in range(1, 6)],
            ]
        )
        trace_writer.writerow(["seed", "variant", "evaluations", "best"])
        for seed in range(30):
            start_stream, evolution_stream = np.random.SeedSequence(seed).spawn(2)
            initial = initial_population(
                np.random.default_rng(start_stream), demands, capacities
            )
            assert np.all(initial @ demands.T <= capacities)
            order = ["repair", "ranking"] if seed % 2 == 0 else ["ranking", "repair"]
            for variant in order:
                random_generator = np.random.default_rng(evolution_stream)
                start = time.perf_counter()
                selection, trace, infeasible_count, removal_count = solve(
                    initial, random_generator, utilities, demands, capacities, variant
                )
                seconds = time.perf_counter() - start
                runs_writer.writerow(
                    [
                        seed,
                        variant,
                        BUDGET,
                        trace[-1][1],
                        seconds,
                        infeasible_count,
                        BUDGET - POPULATION,
                        removal_count,
                        " ".join(map(str, np.flatnonzero(selection) + 1)),
                        *map(int, demands[:, selection].sum(axis=1)),
                    ]
                )
                trace_writer.writerows(
                    (seed, variant, evaluation_count, best_utility)
                    for evaluation_count, best_utility in trace
                )


# Statistical analysis and figures

def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))

def analyse_results():
    _, _, utilities, demands, capacities = load_workload_instance(
        ROOT / "multi_resource_n90_m5.txt"
    )
    rows = read(RESULTS / "challenge_runs.csv")
    traces = read(RESULTS / "challenge_trace.csv")
    traces_by_run = defaultdict(list)
    for trace in traces:
        traces_by_run[(trace["variant"], trace["seed"])].append(trace)
    summary, scores = {}, {}
    figure, (quality, trend) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ["#00796b", "#a95517"]
    for pos, (variant, color) in enumerate(zip(("repair", "ranking"), colors), start=1):
        group = sorted(
            (run for run in rows if run["variant"] == variant),
            key=lambda run: int(run["seed"]),
        )
        assert [int(run["seed"]) for run in group] == list(range(30))
        curves = []
        for run in group:
            model_indices = [int(j) - 1 for j in run["models_1_based"].split()]
            assert len(set(model_indices)) == len(model_indices) and all(
                0 <= j < len(utilities) for j in model_indices
            )
            usage = demands[:, model_indices].sum(axis=1)
            assert np.all(usage <= capacities) and int(
                utilities[model_indices].sum()
            ) == int(run["best"])
            assert list(usage) == [int(run[f"usage_{i}"]) for i in range(1, 6)]
            trace = traces_by_run[(variant, run["seed"])]
            grid = [int(checkpoint["evaluations"]) for checkpoint in trace]
            curve = [int(checkpoint["best"]) for checkpoint in trace]
            assert (
                grid == list(range(80, 20001, 80))
                and int(run["evaluations"]) == grid[-1]
            )
            assert min(np.diff(curve)) >= 0 and curve[-1] == int(
                run["best"]
            )
            curves.append(curve)
        final_utilities = np.array([int(run["best"]) for run in group])
        scores[variant] = final_utilities
        invalid = sum(int(run["invalid_children"]) for run in group)
        removed = sum(int(run["removed"]) for run in group)
        summary[variant] = dict(
            quantiles=np.percentile(final_utilities, [0, 25, 50, 75, 100]).tolist(),
            median_seconds=float(
                np.median([float(run["search_seconds"]) for run in group])
            ),
            invalid_rate=invalid / sum(int(run["offspring"]) for run in group),
            removed=removed,
            removals_per_invalid=removed / invalid,
            best_run=group[final_utilities.argmax()],
        )
        quality.boxplot(
            final_utilities,
            positions=[pos],
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.3},
            medianprops={"color": color},
        )
        low, median, high = np.percentile(curves, [25, 50, 75], axis=0)
        trend.plot(grid, median, label=variant.title(), color=color)
        trend.fill_between(grid, low, high, color=color, alpha=0.16)
    for repair_utility, ranking_utility in zip(scores["repair"], scores["ranking"]):
        quality.plot(
            [1, 2],
            [repair_utility, ranking_utility],
            color="#465772",
            alpha=0.18,
            lw=0.75,
        )
    quality.set(
        xticks=[1, 2],
        xticklabels=["Repair", "Ranking"],
        ylabel="Final feasible utility",
        title="Paired endpoints",
    )
    trend.set(
        xlabel="Candidate evaluations",
        ylabel="Best feasible utility",
        title="Median and middle 50%",
    )
    trend.legend()
    figure.savefig(RESULTS / "challenge.png", dpi=180)
    paired_differences = scores["repair"] - scores["ranking"]
    wilcoxon_result = wilcoxon(
        paired_differences, method="asymptotic", alternative="two-sided"
    )
    wins, losses = int((paired_differences > 0).sum()), int(
        (paired_differences < 0).sum()
    )
    summary["comparison"] = dict(
        median_difference=float(np.median(paired_differences)),
        wins=wins,
        losses=losses,
        ties=int((paired_differences == 0).sum()),
        wilcoxon_W=float(wilcoxon_result.statistic),
        wilcoxon_p=float(wilcoxon_result.pvalue),
        sign_p=binomtest(wins, wins + losses).pvalue,
    )
    # Optimistic quality bound for interpretation only. No search can access this result.
    relaxation = linprog(
        -utilities, A_ub=demands, b_ub=capacities, bounds=(0, 1), method="highs"
    )
    assert relaxation.success
    summary["lp_upper_bound"] = -relaxation.fun
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

def main():
    print("Part 1: running 30 hill-climber experiments")
    run_baseline()
    print("\nParts 2 and 3: running 30 paired genetic algorithm experiments")
    run_experiments()
    print("\nValidating results and producing statistical analysis")
    analyse_results()
    print("\nCompleted all 90 runs. Results and figures are saved beside this script.")


if __name__ == "__main__":
    main()
