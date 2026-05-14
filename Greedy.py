import pickle
from typing import Dict, List

from order import read_order, total
from decoder import Item, read, decode_one_bin_ep_order_list, density_fitness


BIN_SIZE = (1000, 1400, 1400)
BIN_VOL = BIN_SIZE[0] * BIN_SIZE[1] * BIN_SIZE[2]

vol = {
    1: 5322948, 2: 19635000, 3: 28210000, 4: 27300000, 5: 18427500,
    6: 10920000, 7: 18154500, 8: 12675000, 9: 2925000, 10: 12675000,
}

box_type_items = read("box.csv")
BOX_TYPES: Dict[int, Item] = {it.id: it for it in box_type_items}

orders = read_order("single_order.csv")
all_orders = total(orders)


def expand_orders(order_amounts: Dict[int, int]) -> List[int]:

    full_order: List[int] = []
    for type_id, count in order_amounts.items():
        full_order.extend([type_id] * count)
    return full_order


def greedy_sort_desc(order_list: List[int]) -> List[int]:

    #Descending greedy

    return sorted(order_list, key=lambda t: (-vol[t], t))


def extract_successful_order(full_order: List[int], placements: List) -> List[int]:

    placed_item_ids = sorted(p.item_id for p in placements)
    return [full_order[item_id - 1] for item_id in placed_item_ids]


def main() -> List[list]:
    test_order_all = expand_orders(all_orders)
    print("Initial order:", test_order_all)
    print("Total boxes:", len(test_order_all))

    finished: List[list] = []
    bin_idx = 0

    while test_order_all:
        print("remaining:", len(test_order_all))


        greedy_order = greedy_sort_desc(test_order_all)

        placements, failed_type_ids, _ = decode_one_bin_ep_order_list(greedy_order)
        placed_order = extract_successful_order(greedy_order, placements)

        if not placed_order:
            raise RuntimeError(
                "Decoder failed to place any box in the current round. "
            )

        final_order = placed_order
        final_placements = placements
        final_fitness = density_fitness(final_placements, final_order) if final_order else 0.0

        bin_idx += 1
        print(
            f"Bin {bin_idx}: "
            f"boxes={len(final_order)}, "
            f"failed={len(failed_type_ids)}, "
            f"fitness={final_fitness:.6f}"
        )

        finished.append([
            final_order,
            [],
            final_fitness,
            final_placements,
        ])

        test_order_all = failed_type_ids

    with open("finished_results_greedy.pkl", "wb") as f:
        pickle.dump(finished, f)

    return finished


if __name__ == "__main__":
    main()