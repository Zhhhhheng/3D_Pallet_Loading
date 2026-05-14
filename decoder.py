import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

#designed based on extreme point method from Crainic et al., 2008, "Extreme Point-Based Heuristics for Three-Dimensional Bin Packing"

#  Data structures
@dataclass
class Item:         #information of boxes
    id: int
    l: int
    w: int
    h: int
    type_id: int


@dataclass
class Placement:     #information of placements(start point + end points + orientation )
    item_id: int
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int
    ori: int  # 0..5


#  Read CSV
def read(file_path: str) -> List[Item]:
    items: List[Item] = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(Item(
                id=int(row["NUMBER"]),
                l=int(row["LENGTH"]),
                w=int(row["WIDTH"]),
                h=int(row["HEIGHT"]),
                type_id=int(row["NUMBER"])
            ))
    return items


BIN_SIZE = (1000, 1400, 1400)

box_type_items = read("box.csv")
BOX_TYPES = {it.id: it for it in box_type_items}

def packing_height(placements: List["Placement"]) -> int:

    if not placements:
        return 0
    return max(p.z + p.dz for p in placements)  #packing height = highest Z


def density_fitness(
    placements: List["Placement"],
    placed_type_ids: List[int],
    bin_size: Tuple[int, int, int] = BIN_SIZE,
) -> float:                                                  #calculate the density

    if not placements or not placed_type_ids:
        return 0.0

    H = packing_height(placements)
    if H == 0:
        return 0.0

    vol_map = {
        1: 5322948, 2: 19635000, 3: 28210000, 4: 27300000, 5: 18427500,
        6: 10920000, 7: 18154500, 8: 12675000, 9: 2925000, 10: 12675000,
    }
    L, W, _ = bin_size
    placed_vol = sum(vol_map[t] for t in placed_type_ids)
    return placed_vol / (L * W * H)

def check_overlap(a: Placement, b: Placement) -> bool:                   #True if two axis-aligned boxes overlap in volume.
    return (
        a.x < b.x + b.dx and b.x < a.x + a.dx and
        a.y < b.y + b.dy and b.y < a.y + a.dy and
        a.z < b.z + b.dz and b.z < a.z + a.dz
    )


def check_fit(x: int, y: int, z: int, dx: int, dy: int, dz: int,
              L: int, W: int, H: int) -> bool:                           #True if a box placed at (x,y,z) with size (dx,dy,dz) is inside the bin
    return (
        0 <= x and 0 <= y and 0 <= z and
        x + dx <= L and y + dy <= W and z + dz <= H
    )


def check_point_inside_box(px: int, py: int, pz: int, b: Placement) -> bool: #True if point (px,py,pz) lies inside the volume of box b. For ep points ONLY!
    return (
        b.x <= px < b.x + b.dx and
        b.y <= py < b.y + b.dy and
        b.z <= pz < b.z + b.dz
    )


#Orientation helper
def all_orientations(l: int, w: int, h: int) -> List[Tuple[int, int, int]]:

    return [
        (l, w, h),
        (l, h, w),
        (w, l, h),
        (w, h, l),
        (h, l, w),
        (h, w, l),
    ]


def contact_area(cand: Placement, placed: List[Placement]) -> int:

    area = 0
    if cand.z == 0:
        area += cand.dx * cand.dy

    for b in placed:
        # bottom touches top
        if cand.z == b.z + b.dz:
            ox = max(0, min(cand.x + cand.dx, b.x + b.dx) - max(cand.x, b.x))
            oy = max(0, min(cand.y + cand.dy, b.y + b.dy) - max(cand.y, b.y))
            area += ox * oy

        # left/right
        if cand.x == b.x + b.dx or cand.x + cand.dx == b.x:
            oy = max(0, min(cand.y + cand.dy, b.y + b.dy) - max(cand.y, b.y))
            oz = max(0, min(cand.z + cand.dz, b.z + b.dz) - max(cand.z, b.z))
            area += oy * oz

        # back/front
        if cand.y == b.y + b.dy or cand.y + cand.dy == b.y:
            ox = max(0, min(cand.x + cand.dx, b.x + b.dx) - max(cand.x, b.x))
            oz = max(0, min(cand.z + cand.dz, b.z + b.dz) - max(cand.z, b.z))
            area += ox * oz

    return area



def placement_score(cand: Placement, placed: List[Placement], alpha: float = 0.1) -> Tuple[float, int, int, int]:

    c = contact_area(cand, placed)
    wall_touch = (1 if cand.x == 0 else 0) + (1 if cand.y == 0 else 0) + (1 if cand.z == 0 else 0)

    return (cand.z - alpha * c, -wall_touch, cand.y, cand.x)


def update_ep(ep: Set[Tuple[int, int, int]],
              new_box: Placement,
              placed: List[Placement],
              L: int, W: int, H: int) -> Set[Tuple[int, int, int]]:

    x, y, z = new_box.x, new_box.y, new_box.z
    dx, dy, dz = new_box.dx, new_box.dy, new_box.dz

    # add new candidate points
    ep.add((x + dx, y, z))
    ep.add((x, y + dy, z))
    ep.add((x, y, z + dz))

    # keep in bounds
    ep2: Set[Tuple[int, int, int]] = set()
    for px, py, pz in ep:
        if 0 <= px <= L and 0 <= py <= W and 0 <= pz <= H:
            ep2.add((px, py, pz))

    # remove points inside any placed box
    ep3: Set[Tuple[int, int, int]] = set()
    for px, py, pz in ep2:
        inside = False
        for b in placed:
            if check_point_inside_box(px, py, pz, b):
                inside = True
                break
        if not inside:
            ep3.add((px, py, pz))

    return ep3

def decode_one_bin_ep(items: List[Item],
                      order: List[int],
                      bin_size: Tuple[int, int, int],
                      alpha: float = 0.1) -> Tuple[List[Placement], List[int]]:

    L, W, H = bin_size
    items_by_id: Dict[int, Item] = {it.id: it for it in items}

    placed: List[Placement] = []
    ep: Set[Tuple[int, int, int]] = {(0, 0, 0)}

    for item_id in order:
        item = items_by_id[item_id]
        best: Optional[Placement] = None
        best_sc: Optional[Tuple[float, int, int, int]] = None

        # try EP points in low-first order to speed up
        ep_list = sorted(ep, key=lambda p: (p[2], p[1], p[0]))
        dims_list = all_orientations(item.l, item.w, item.h)

        for (px, py, pz) in ep_list:
            for ori, (dx, dy, dz) in enumerate(dims_list):
                if not check_fit(px, py, pz, dx, dy, dz, L, W, H):
                    continue

                cand = Placement(item_id=item_id, x=px, y=py, z=pz, dx=dx, dy=dy, dz=dz, ori=ori)
                if any(check_overlap(cand, b) for b in placed):
                    continue

                sc = placement_score(cand, placed, alpha=alpha)
                if best is None or sc < best_sc:
                    best = cand
                    best_sc = sc

        if best is None:
            continue

        placed.append(best)
        ep = update_ep(ep, best, placed, L, W, H)

    placed_ids = {p.item_id for p in placed}
    failed = [i for i in order if i not in placed_ids]
    return placed, failed

def decode_one_bin_ep_order_list(order_list, alpha: float = 0.1):
    items = []
    for i, type_id in enumerate(order_list):
        t = BOX_TYPES[type_id]
        items.append(Item(
            id=i + 1,
            l=t.l,
            w=t.w,
            h=t.h,
            type_id=type_id
        ))

    order = list(range(1, len(items) + 1))
    placements, failed_item_ids = decode_one_bin_ep(items, order, BIN_SIZE, alpha=alpha)
    failed_type_ids = [order_list[i - 1] for i in failed_item_ids]

    return placements, failed_type_ids, failed_item_ids
