import csv
import random



def read_order(file_path):
    orders = []

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            order_id = int(row["order_id"])

            amounts = []
            for i in range(1, 11):
                amounts.append(int(row[f"amt_{i}"]))

            orders.append({
                "order_id": order_id,
                "amounts": amounts
            })

    return orders


orders = read_order("order.csv")



def total(orders):
    all_order = {}
    for n in range(1, 11):
        all_order[n] = 0

    for i in orders:
        l = i["amounts"]
        for a, count in enumerate(l):
            all_order[a + 1] = all_order[a + 1] + count

    return all_order
