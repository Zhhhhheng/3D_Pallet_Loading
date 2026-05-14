import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from decoder import Item, decode_one_bin_ep, density_fitness, read


POP_SIZE = 300
GENERATIONS = 30
BIN_SIZE = (1000, 1400, 1400)
BIN_VOL = BIN_SIZE[0] * BIN_SIZE[1] * BIN_SIZE[2]

vol = {
    1: 5322948, 2: 19635000, 3: 28210000, 4: 27300000, 5: 18427500,
    6: 10920000, 7: 18154500, 8: 12675000, 9: 2925000, 10: 12675000,
}

box_type_items = read("box.csv")
BOX_TYPES: Dict[int, Item] = {it.id: it for it in box_type_items}
_fitness_cache: Dict[Tuple[int, ...], float] = {}

def _evaluate_tuple(order_tuple: Tuple[int, ...]) -> float:         #for multiprocessing

    items: List[Item] = []
    for i, type_id in enumerate(order_tuple):
        t = BOX_TYPES[type_id]
        items.append(Item(
            id=i + 1,
            l=t.l,
            w=t.w,
            h=t.h,
            type_id=type_id,
        ))

    order = list(range(1, len(items) + 1))
    placements, failed_item_ids = decode_one_bin_ep(items, order, BIN_SIZE)

    placed_item_ids_set = {p.item_id for p in placements}
    placed_type_ids = [
        order_tuple[i - 1]
        for i in range(1, len(order_tuple) + 1)
        if i in placed_item_ids_set
    ]
    return density_fitness(placements, placed_type_ids, BIN_SIZE)

def evaluate_population_parallel(
    individuals: Iterable[List[int]],
    max_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Dict[Tuple[int, ...], float]:

    keys = [tuple(ind) for ind in individuals]
    missing_keys = list(dict.fromkeys(k for k in keys if k not in _fitness_cache))

    if missing_keys:
        if use_parallel and (max_workers is None or max_workers != 1):
            workers = max_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for key, fit in zip(missing_keys, executor.map(_evaluate_tuple, missing_keys)):
                    _fitness_cache[key] = fit
        else:
            for key in missing_keys:
                _fitness_cache[key] = _evaluate_tuple(key)

    return {key: _fitness_cache[key] for key in keys}

def initialization(
    base_order: List[int],
    pop_size: int = POP_SIZE,
    keep_base: bool = True,
) -> List[List[int]]:
    population: List[List[int]] = []

    if keep_base:
        population.append(base_order.copy())

    while len(population) < pop_size:
        individual = base_order.copy()
        random.shuffle(individual)
        population.append(individual)

    return population


def crossover_count_preserving(
    parent1: List[int],
    parent2: List[int],
    base_counts: Counter,
    mean_ratio: float = 0.20,
    std_ratio: float = 0.10,
) -> List[int]:
    """
    Count-preserving crossover with adaptive segment length.
    """
    if len(parent1) != len(parent2):         #two parents must have same length
        raise ValueError("parent1 and parent2 must have the same length")

    n = len(parent1)
    if n == 0:
        return []

    cut_length = int(round(np.random.normal(mean_ratio * n, std_ratio * n)))
    cut_length = max(1, min(cut_length, n))

    start = random.randint(0, n - cut_length)  #ramdlomly choose start point
    end = start + cut_length

    child: List[int] = [None] * n  # type: ignore
    child[start:end] = parent1[start:end]

    remaining = dict(base_counts)
    for x in child[start:end]:
        remaining[x] -= 1        #calculate the counts for each type

    fill_positions = list(range(0, start)) + list(range(end, n))
    p2_iter = iter(parent2)

    for pos in fill_positions:
        gene = None

        while True:
            g = next(p2_iter, None)
            if g is None:
                break
            if remaining.get(g, 0) > 0:
                gene = g
                break

        if gene is None:
            for k, v in remaining.items():
                if v > 0:
                    gene = k
                    break

        child[pos] = gene
        remaining[gene] -= 1

    return child


def parent_selection(
    population: List[List[int]],
    fitness_map: Dict[Tuple[int, ...], float],
    num_parents: int = 30,
    tournament_size: int = 5,
) -> List[List[int]]:
    parents: List[List[int]] = []
    for _ in range(num_parents):
        competitors = random.sample(population, tournament_size)
        winner = max(competitors, key=lambda x: fitness_map[tuple(x)])
        parents.append(winner)
    return parents


def make_children(
    parents: List[List[int]],
    num_children: int,
    base_counts: Counter,
    crossover_mean_ratio: float = 0.20,
    crossover_std_ratio: float = 0.10,
) -> List[List[int]]:
    children: List[List[int]] = []
    while len(children) < num_children:
        p1, p2 = random.sample(parents, 2)
        child = crossover_count_preserving(
            p1,
            p2,
            base_counts,
            mean_ratio=crossover_mean_ratio,
            std_ratio=crossover_std_ratio,
        )
        children.append(child)
    return children


def child_selection(
    population: List[List[int]],
    children: List[List[int]],
    fitness_map: Dict[Tuple[int, ...], float],
    pop_size: int = POP_SIZE,
) -> List[List[int]]:
    combined = population + children
    combined_sorted = sorted(combined, key=lambda x: fitness_map[tuple(x)], reverse=True)
    return combined_sorted[:pop_size]

def decode_result(best_order: List[int]) -> Tuple[List, List[int], List[int]]:

    items: List[Item] = []
    for i, type_id in enumerate(best_order):
        t = BOX_TYPES[type_id]
        items.append(Item(i + 1, t.l, t.w, t.h, type_id))

    order = list(range(1, len(items) + 1))
    placements, failed_item_ids = decode_one_bin_ep(items, order, BIN_SIZE)
    failed_type_ids = [best_order[i - 1] for i in failed_item_ids]
    return placements, failed_item_ids, failed_type_ids

def initialize_population_and_fitness(
    base_order: List[int],
    pop_size: int,
    keep_base: bool,
    max_workers: Optional[int],
    use_parallel: bool,
) -> Tuple[List[List[int]], Dict[Tuple[int, ...], float], Counter]:

    _fitness_cache.clear()

    base_counts = Counter(base_order)
    population = initialization(base_order, pop_size=pop_size, keep_base=keep_base)
    population_fitness = evaluate_population_parallel(
        population,
        max_workers=max_workers,
        use_parallel=use_parallel,
    )
    return population, population_fitness, base_counts


def run_one_generation(
    population: List[List[int]],
    population_fitness: Dict[Tuple[int, ...], float],
    base_counts: Counter,
    pop_size: int,
    num_parents: int,
    tournament_size: int,
    max_workers: Optional[int],
    use_parallel: bool,
    crossover_mean_ratio: float,
    crossover_std_ratio: float,
) -> Tuple[List[List[int]], Dict[Tuple[int, ...], float]]:

    parents = parent_selection(
        population,
        population_fitness,
        num_parents=num_parents,
        tournament_size=tournament_size,
    )

    children = make_children(
        parents,
        num_children=pop_size,
        base_counts=base_counts,
        crossover_mean_ratio=crossover_mean_ratio,
        crossover_std_ratio=crossover_std_ratio,
    )

    combined = population + children
    combined_fitness = evaluate_population_parallel(
        combined,
        max_workers=max_workers,
        use_parallel=use_parallel,
    )

    population = child_selection(
        population,
        children,
        combined_fitness,
        pop_size=pop_size,
    )
    population_fitness = {tuple(ind): combined_fitness[tuple(ind)] for ind in population}

    return population, population_fitness


def select_best_individual(
    population: List[List[int]],
    population_fitness: Dict[Tuple[int, ...], float],
    base_counts: Counter,
) -> Tuple[List[int], float]:

    best = max(population, key=lambda ind: population_fitness[tuple(ind)])
    best_fit = population_fitness[tuple(best)]

    assert Counter(best) == base_counts, "Best order counts changed!"
    return best, best_fit


def optimize_packing_sequence(
    base_order: List[int],
    generations: int = GENERATIONS,
    pop_size: int = POP_SIZE,
    num_parents: int = 30,
    tournament_size: int = 5,
    keep_base: bool = True,
    verbose: bool = True,
    max_workers: Optional[int] = None,
    use_parallel: bool = True,
    crossover_mean_ratio: float = 0.20,
    crossover_std_ratio: float = 0.10,
) -> Tuple[List[int], float]:

    population, population_fitness, base_counts = initialize_population_and_fitness(
        base_order=base_order,
        pop_size=pop_size,
        keep_base=keep_base,
        max_workers=max_workers,
        use_parallel=use_parallel,
    )

    best_overall: Optional[List[int]] = None
    best_overall_fit = float("-inf")

    for gen in range(generations):
        if verbose:
            print(f"\n--- Generation {gen + 1} ---")

        population, population_fitness = run_one_generation(
            population=population,
            population_fitness=population_fitness,
            base_counts=base_counts,
            pop_size=pop_size,
            num_parents=num_parents,
            tournament_size=tournament_size,
            max_workers=max_workers,
            use_parallel=use_parallel,
            crossover_mean_ratio=crossover_mean_ratio,
            crossover_std_ratio=crossover_std_ratio,
        )

        best, best_fit = select_best_individual(
            population=population,
            population_fitness=population_fitness,
            base_counts=base_counts,
        )

        if verbose:
            workers = max_workers or os.cpu_count() or 1
            mode = "parallel" if use_parallel and workers != 1 else "single-core"
            print(f"Best fitness: {best_fit}")

        if best_fit > best_overall_fit:
            best_overall = best.copy()
            best_overall_fit = best_fit

    if best_overall is None:
        raise RuntimeError("GA failed to produce any valid individual.")

    return best_overall, best_overall_fit


def run_ga(
    base_order: List[int],
    generations: int = GENERATIONS,
    pop_size: int = POP_SIZE,
    num_parents: int = 30,
    tournament_size: int = 5,
    keep_base: bool = True,
    verbose: bool = True,
    max_workers: Optional[int] = None,
    use_parallel: bool = True,
    crossover_mean_ratio: float = 0.20,
    crossover_std_ratio: float = 0.10,
):
    best_order, best_fit = optimize_packing_sequence(
        base_order=base_order,
        generations=generations,
        pop_size=pop_size,
        num_parents=num_parents,
        tournament_size=tournament_size,
        keep_base=keep_base,
        verbose=verbose,
        max_workers=max_workers,
        use_parallel=use_parallel,
        crossover_mean_ratio=crossover_mean_ratio,
        crossover_std_ratio=crossover_std_ratio,
    )

    placements, failed_item_ids, failed_type_ids = decode_result(best_order)  
    return best_order, failed_type_ids, best_fit, placements
