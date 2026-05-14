import os
import pickle
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from decoder import Item, decode_one_bin_ep_order_list, density_fitness, read
from generate import all_random
from order import read_order, total

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

orders = read_order("order.csv")
ALL_ORDERS = total(orders)

Chromosome = List[List[int]]

# fitness is a single scalar: sum(V_i placed) / (N * BIN_VOL)
GlobalFitness = float
LexFitness = GlobalFitness          # alias kept so call-sites need no change
_fitness_cache: Dict[Tuple[Tuple[int, ...], ...], GlobalFitness] = {}



def flatten_bins(chromosome: Chromosome) -> List[int]:
    flat: List[int] = []
    for pallet in chromosome:
        flat.extend(pallet)
    return flat


def chromosome_key(chromosome: Chromosome) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(p) for p in chromosome)


def total_counter(chromosome: Chromosome) -> Counter:
    return Counter(flatten_bins(chromosome))


def extract_successful_order(full_order: List[int], placements: List) -> List[int]:
    placed_item_ids = sorted(p.item_id for p in placements)
    return [full_order[item_id - 1] for item_id in placed_item_ids]


def repair_empty_bins(chromosome: Chromosome) -> Chromosome:
    return [p for p in chromosome if p]


def random_partition_sizes(total_len: int, target_bins: int) -> List[int]:
    if total_len <= 0:
        return []
    target_bins = max(1, min(target_bins, total_len))
    if target_bins == 1:
        return [total_len]
    cut_points = sorted(random.sample(range(1, total_len), target_bins - 1))
    points = [0] + cut_points + [total_len]
    return [points[i + 1] - points[i] for i in range(len(points) - 1)]


def estimate_bin_count(order_list: List[int]) -> int:
    total_volume = sum(vol[t] for t in order_list)
    rough = max(1, int(round(total_volume / BIN_VOL)))
    return rough


def repartition_sequence(sequence: List[int], sizes: List[int]) -> Chromosome:
    bins: Chromosome = []
    idx = 0
    for size in sizes:
        if idx >= len(sequence):
            break
        bins.append(sequence[idx: idx + size])
        idx += size
    if idx < len(sequence):
        if bins:
            bins[-1].extend(sequence[idx:])
        else:
            bins.append(sequence[idx:])
    return repair_empty_bins(bins)


def lexicographic_fitness(densities: List[float]) -> GlobalFitness:
    N = len(densities)
    if N == 0:
        return 0.0

    return sum(densities) / N



def simulate_chromosome(chromosome: Chromosome) -> Tuple[LexFitness, List[list]]:

    chromosome = repair_empty_bins(chromosome)
    original_counter = total_counter(chromosome)

    finished: List[list] = []
    densities: List[float] = []
    carry_over: List[int] = []

    # Planned pallets
    for planned_pallet in chromosome:
        current_order = carry_over + planned_pallet
        placements, failed_type_ids, _ = decode_one_bin_ep_order_list(current_order)
        placed_order = extract_successful_order(current_order, placements)
        current_density = density_fitness(placements, placed_order, BIN_SIZE)

        finished.append([placed_order, [], current_density, placements])
        densities.append(current_density)
        carry_over = failed_type_ids

    while carry_over:
        current_order = carry_over
        placements, failed_type_ids, _ = decode_one_bin_ep_order_list(current_order)
        placed_order = extract_successful_order(current_order, placements)

        if not placed_order:
            next_round: List[int] = []
            progress = False
            for t in carry_over:
                placements_s, failed_single, _ = decode_one_bin_ep_order_list([t])
                placed_single = extract_successful_order([t], placements_s)
                if placed_single:
                    d = density_fitness(placements_s, placed_single, BIN_SIZE)
                    finished.append([placed_single, [], d, placements_s])
                    densities.append(d)
                    progress = True
                else:
                    next_round.extend(failed_single or [t])

            carry_over = next_round
            continue

        current_density = density_fitness(placements, placed_order, BIN_SIZE)
        finished.append([placed_order, [], current_density, placements])
        densities.append(current_density)
        carry_over = failed_type_ids

    placed_all: List[int] = []
    for best_order, _, _, _ in finished:
        placed_all.extend(best_order)
    placed_counter = Counter(placed_all)
    if placed_counter != original_counter:
        raise RuntimeError(
            "Box conservation check failed.\n"
            f"Missing: {original_counter - placed_counter}\n"
            f"Extra:   {placed_counter - original_counter}"
        )

    if not finished:
        return 0.0, finished

    fitness = lexicographic_fitness(densities)
    return fitness, finished


def _evaluate_key(key: Tuple[Tuple[int, ...], ...]) -> LexFitness:
    chromosome = [list(p) for p in key]
    fitness, _ = simulate_chromosome(chromosome)
    return fitness

def evaluate_population_parallel(
    individuals: Iterable[Chromosome],
    max_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Dict[Tuple[Tuple[int, ...], ...], LexFitness]:

    keys = [chromosome_key(ind) for ind in individuals]
    missing_keys = list(dict.fromkeys(k for k in keys if k not in _fitness_cache))

    if missing_keys:
        if use_parallel and (max_workers is None or max_workers != 1):
            workers = max_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for key, fit in zip(missing_keys, executor.map(_evaluate_key, missing_keys)):
                    _fitness_cache[key] = fit
        else:
            for key in missing_keys:
                _fitness_cache[key] = _evaluate_key(key)

    return {key: _fitness_cache[key] for key in keys}

def initialization(
    base_order: List[int],
    pop_size: int = POP_SIZE,
    keep_base: bool = True,
) -> List[Chromosome]:
    population: List[Chromosome] = []
    n = len(base_order)
    base_bin_count = estimate_bin_count(base_order)

    if keep_base:
        sizes = random_partition_sizes(n, base_bin_count)
        population.append(repartition_sequence(base_order.copy(), sizes))

    while len(population) < pop_size:
        shuffled = base_order.copy()
        random.shuffle(shuffled)
        target_bins = max(1, base_bin_count + random.choice([-2, -1, 0, 1, 2]))
        target_bins = min(target_bins, n)
        sizes = random_partition_sizes(n, target_bins)
        candidate = repartition_sequence(shuffled, sizes)
        population.append(candidate)

    return population


def crossover_count_preserving_sequence(
    parent1: List[int],
    parent2: List[int],
    base_counts: Counter,
    mean_ratio: float = 0.20,
    std_ratio: float = 0.1,
) -> List[int]:
    if len(parent1) != len(parent2):
        raise ValueError("parent1 and parent2 must have the same length")
    n = len(parent1)
    if n == 0:
        return []

    cut_length = int(round(np.random.normal(mean_ratio * n, std_ratio * n)))
    cut_length = max(1, min(cut_length, n))
    start = random.randint(0, n - cut_length)
    end = start + cut_length

    child: List[Optional[int]] = [None] * n
    child[start:end] = parent1[start:end]

    remaining = dict(base_counts)
    for x in child[start:end]:
        if x is not None:
            remaining[x] -= 1

    fill_positions = list(range(0, start)) + list(range(end, n))
    p2_iter = iter(parent2)

    for pos in fill_positions:
        gene: Optional[int] = None
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
        if gene is None:
            raise RuntimeError("Failed to fill child chromosome while preserving counts.")
        child[pos] = gene
        remaining[gene] -= 1

    return [int(x) for x in child]

def crossover_with_boundaries(
    parent1: Chromosome,
    parent2: Chromosome,
    base_counts: Counter,
    mean_ratio: float = 0.20,
    std_ratio: float = 0.10,
) -> Chromosome:
    seq1 = flatten_bins(parent1)
    seq2 = flatten_bins(parent2)
    child_seq = crossover_count_preserving_sequence(
        seq1, seq2, base_counts, mean_ratio=mean_ratio, std_ratio=std_ratio
    )

    donor = parent1 if random.random() < 0.5 else parent2
    donor_sizes = [len(p) for p in donor]

    if donor_sizes and random.random() < 0.35:
        j = random.randrange(len(donor_sizes))
        donor_sizes[j] += random.choice([-1, 1])

    donor_sizes = [s for s in donor_sizes if s > 0]
    if not donor_sizes:
        donor_sizes = [len(child_seq)]

    total_sz = sum(donor_sizes)
    if total_sz != len(child_seq):
        factor = len(child_seq) / total_sz
        donor_sizes = [max(1, int(round(s * factor))) for s in donor_sizes]
        diff = len(child_seq) - sum(donor_sizes)
        donor_sizes[-1] += diff
        donor_sizes = [max(1, s) for s in donor_sizes]
        while sum(donor_sizes) > len(child_seq):
            for i in range(len(donor_sizes) - 1, -1, -1):
                if donor_sizes[i] > 1:
                    donor_sizes[i] -= 1
                    break

    cursor, adjusted, remaining = 0, [], len(child_seq)
    for k, sz in enumerate(donor_sizes):
        pallets_left = len(donor_sizes) - k
        if pallets_left == 1:
            adjusted.append(remaining)
            break
        window = child_seq[cursor: cursor + sz]
        donor_set = Counter(donor[k])
        overlap = sum(min(donor_set[b], window.count(b)) for b in donor_set)
        new_sz = max(1, int(round((overlap / sz if sz else 1.0) * sz)))
        new_sz = min(new_sz, remaining - (pallets_left - 1))
        adjusted.append(max(1, new_sz))
        cursor += adjusted[-1]
        remaining -= adjusted[-1]
    donor_sizes = adjusted

    return repartition_sequence(child_seq, donor_sizes)


def mutate_boundaries(
    chromosome: Chromosome,
    mutation_rate: float = 0.25,
) -> Chromosome:
    child = [p.copy() for p in chromosome]
    if random.random() > mutation_rate or not child:
        return repair_empty_bins(child)

    op = random.choice(["swap_within", "move_across", "swap_across", "split", "merge"])

    if op == "swap_within":
        bins = [i for i, p in enumerate(child) if len(p) >= 2]
        if bins:
            b = random.choice(bins)
            i, j = random.sample(range(len(child[b])), 2)
            child[b][i], child[b][j] = child[b][j], child[b][i]

    elif op == "move_across":
        src_bins = [i for i, p in enumerate(child) if len(p) >= 1]
        if src_bins and len(child) >= 2:
            src = random.choice(src_bins)
            dst_candidates = [i for i in range(len(child)) if i != src]
            dst = random.choice(dst_candidates)
            pos = random.randrange(len(child[src]))
            gene = child[src].pop(pos)
            ins = random.randrange(len(child[dst]) + 1)
            child[dst].insert(ins, gene)

    elif op == "swap_across":
        bins = [i for i, p in enumerate(child) if len(p) >= 1]
        if len(bins) >= 2:
            b1, b2 = random.sample(bins, 2)
            i = random.randrange(len(child[b1]))
            j = random.randrange(len(child[b2]))
            child[b1][i], child[b2][j] = child[b2][j], child[b1][i]

    elif op == "split":
        bins = [i for i, p in enumerate(child) if len(p) >= 3]
        if bins:
            b = random.choice(bins)
            cut = random.randint(1, len(child[b]) - 1)
            new_bin = child[b][cut:]
            child[b] = child[b][:cut]
            child.insert(b + 1, new_bin)

    elif op == "merge":
        if len(child) >= 2:
            b = random.randrange(len(child) - 1)
            child[b].extend(child[b + 1])
            del child[b + 1]

    return repair_empty_bins(child)

def parent_selection(
    population: List[Chromosome],
    fitness_map: Dict[Tuple[Tuple[int, ...], ...], LexFitness],
    num_parents: int = 30,
    tournament_size: int = 5,
) -> List[Chromosome]:

    parents: List[Chromosome] = []
    for _ in range(num_parents):
        competitors = random.sample(population, tournament_size)
        winner = max(competitors, key=lambda x: fitness_map[chromosome_key(x)])
        parents.append(winner)
    return parents


def make_children(
    parents: List[Chromosome],
    num_children: int,
    base_counts: Counter,
    crossover_mean_ratio: float = 0.20,
    crossover_std_ratio: float = 0.10,
    mutation_rate: float = 0.24,
) -> List[Chromosome]:
    children: List[Chromosome] = []
    while len(children) < num_children:
        p1, p2 = random.sample(parents, 2)
        child = crossover_with_boundaries(
            p1, p2, base_counts,
            mean_ratio=crossover_mean_ratio,
            std_ratio=crossover_std_ratio,
        )
        child = mutate_boundaries(child, mutation_rate=mutation_rate)
        children.append(child)
    return children


def child_selection(
    population: List[Chromosome],
    children: List[Chromosome],
    fitness_map: Dict[Tuple[Tuple[int, ...], ...], LexFitness],
    pop_size: int = POP_SIZE,
) -> List[Chromosome]:

    combined = population + children
    combined_sorted = sorted(
        combined,
        key=lambda x: fitness_map[chromosome_key(x)],
        reverse=True,
    )
    return combined_sorted[:pop_size]

def run_global_ga_with_boundaries(
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
    crossover_std_ratio: float = 0.1,
    mutation_rate: float = 0.24,
):
    _fitness_cache.clear()

    base_counts = Counter(base_order)
    population = initialization(base_order, pop_size=pop_size, keep_base=keep_base)
    population_fitness = evaluate_population_parallel(
        population, max_workers=max_workers, use_parallel=use_parallel,
    )

    best_overall: Optional[Chromosome] = None
    best_overall_fit: GlobalFitness = 0.0   # sentinel "worst"

    for gen_idx in range(generations):
        parents = parent_selection(
            population, population_fitness,
            num_parents=num_parents, tournament_size=tournament_size,
        )
        children = make_children(
            parents, num_children=pop_size, base_counts=base_counts,
            crossover_mean_ratio=crossover_mean_ratio,
            crossover_std_ratio=crossover_std_ratio,
            mutation_rate=mutation_rate,
        )

        combined = population + children
        combined_fitness = evaluate_population_parallel(
            combined, max_workers=max_workers, use_parallel=use_parallel,
        )
        population = child_selection(population, children, combined_fitness, pop_size=pop_size)
        population_fitness = {
            chromosome_key(ind): combined_fitness[chromosome_key(ind)]
            for ind in population
        }

        best = max(population, key=lambda ind: population_fitness[chromosome_key(ind)])
        best_fit: LexFitness = population_fitness[chromosome_key(best)]

        assert total_counter(best) == base_counts, "Best chromosome counts changed!"

        if verbose:
            workers = max_workers or os.cpu_count() or 1
            mode = "parallel" if use_parallel and workers != 1 else "single-core"
            print(
                f"Generation {gen_idx + 1}/{generations} | "
                f"global density = {best_fit:.4f} | "
                f"{mode}, workers={workers}"
            )

        if best_fit > best_overall_fit:
            best_overall = [p.copy() for p in best]
            best_overall_fit = best_fit

    _, finished = simulate_chromosome(best_overall)
    return best_overall, best_overall_fit, finished

def main() -> List[list]:
    test_order_all = all_random(ALL_ORDERS.copy())
    print("Initial global order length:", len(test_order_all))

    workers = os.cpu_count() or 1

    best_chromosome, best_global_fitness, finished = run_global_ga_with_boundaries(
        test_order_all,
        generations=30,
        pop_size=300,
        num_parents=30,
        tournament_size=5,
        max_workers=workers,
        use_parallel=True,
        verbose=True,
        crossover_mean_ratio=0.20,
        crossover_std_ratio=0.1,
        mutation_rate=0.22,
    )

    print(f"\nBest global density fitness: {best_global_fitness:.4f}")
    print(f"Planned pallets in chromosome : {len(best_chromosome)}")
    print(f"Final pallets after decode    : {len(finished)}")

    with open("finished_results_global_direct_boundaries_lex.pkl", "wb") as f:
        pickle.dump(finished, f)

    with open("best_global_chromosome_direct_boundaries_lex.pkl", "wb") as f:
        pickle.dump(
            {
                "best_chromosome": best_chromosome,
                "best_fitness": best_global_fitness,
                "fitness_definition": (
                    "Method B — global volumetric density (scalar). "
                    "f = mean(density_k for k in 1..N) "
                    "  = total_placed_volume / (N * BIN_VOL). "
                    "Higher is better: fewer pallets and denser packing "
                    "both increase f without any hand-tuned penalty weight. "
                    "density_k = sum(V_i placed on pallet k) / BIN_VOL."
                ),
                "representation": "explicit pallet boundaries, no cutter",
                "pop_size": 300,
                "generations": 30,
                "num_parents": 30,
                "tournament_size": 5,
                "crossover_mean_ratio": 0.20,
                "crossover_std_ratio": 0.1,
                "mutation_rate": 0.22,
            },
            f,
        )

    return finished


if __name__ == "__main__":
    main()