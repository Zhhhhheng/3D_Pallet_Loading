from order import read_order, total
from decoder import Item, read
import random

orders = read_order('order.csv')
all_orders = total(orders)



def all_random(all):
    random_list = []
    total_nr = sum(all.values())

    while len(random_list) < total_nr:
        n = random.randint(1, 10)
        if all[n] > 0:
            random_list.append(n)
            all[n] = all[n] - 1
        if all[n] == 0:
            pass
    return random_list


vol = {}
item = read('box.csv')
for i in item:
    vol[i.id] = i.h * i.l * i.w
pallet_vol = 1000 * 1400 * 1400

def gen (random_list,pallet_vol):
    pallet = []
    r = 0
    for n in random_list:
        r = r + vol[n]
        if r <   pallet_vol:
            pallet.append(n)

            continue

        if r >  pallet_vol:

            break


    return pallet


