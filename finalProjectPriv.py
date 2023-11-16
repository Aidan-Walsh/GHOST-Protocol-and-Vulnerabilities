import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import random

# in context of ghost

# chain quality as adversarial hash power changes
# adversary will try to make its own heaviest chain
# assume no network control (balance attack assumes more network control)
# adversary will publish its created blocks - it will never hide them NOTE: THIS HAS NO SELFISH COMPONENT
# because Ethereum reward uncle blocks, thus, they would only be reducing
# their chance at being heaviest chain
# no point in doing one node for each because it will essentially just turn into
# longest chain because honest node will just mine at tip
# so will simulate hash power among 100 nodes, where each node has equivalent
# hash power
# difficult to simulate incentives, so will use probability: probability (prob) that
# miner stops mining their block, if someone wins for current level, and then
# mines on new block
# can do two ways: adversary just mines on their own chain
# or adversary inserts its own chain into ghost chain and basically behaves
# as honest node
# adversary does not have probability that it selects new node in first case
# will store all nodes in array
# adversaries will focus entirely on their chain or will focus on both the
# adversary chain and heaviest chain


class Node:
    def __init__(self, parent, children, weight, advers):
        self.parent = parent
        self.children = children
        self.weight = weight
        self.advers = advers


class Worker:
    def __init__(self, adversarial, probability, mining_node):
        self.adversarial = adversarial
        self.probability = probability
        self.mining_node = mining_node


Genesis = Node(parent=None, children=[], weight=1, advers=False)


def update_weights(node):
    curr_node = node
    while True: 
      if (curr_node.parent == None): 
          break
      curr_node.parent.weight += 1
      curr_node = curr_node.parent



def update_tip(node): # fix this
    
    curr_node = node 
    while True: 
        if len(curr_node.children) == 0: 
            return curr_node
        max_weight = curr_node.children[0].weight
        max_node = curr_node.children[0]
        for i in range(1, len(curr_node.children)):
          if max_weight < curr_node.children[i].weight:
            max_weight = curr_node.children[i].weight
            max_node = curr_node.children[i]
        curr_node = max_node
            




def chain_quality(node):
    curr_node = node
    good_count = 0.0
    total_count = 0.0
    while True:
        if curr_node.parent == None:
            break
        if (curr_node.advers):
            total_count += 1.0
        else:
            total_count += 1.0
            good_count += 1.0
        curr_node = curr_node.parent
    print(total_count)
    return good_count/total_count
        



lam = 1

total_miners = 100


adversary_prob = 0  # probability that adversary switches over to heaviest chain
# if 0, this is a full private attack, goes to 100


for p in [20, 60, 100]:  # probability that miner changes over to new block
    x = []
    y = []
    for bad_miners in range(0, total_miners + 1):
        x_val = bad_miners
        miners = []
        honest_miners = total_miners - bad_miners
        Genesis = Node(parent=None, children=[], weight=1, advers=False)
        block_count = 1
        tip = Genesis

        adversary_tip = Genesis
        hash_power = 1 / total_miners
        # create good and bad miners
        for i in range(0, total_miners):
            if i < honest_miners:
                miners.append(
                    Worker(adversarial=False, probability=0.0, mining_node=Genesis)
                )
            else:
                miners.append(
                    Worker(adversarial=True, probability=0.0, mining_node=Genesis)
                )
            miners[i].probability = np.random.exponential(1 / ((hash_power) * lam))

        while block_count < 1000:
            miners.sort(key=lambda x: x.probability)
            min_prob = 0
            adversary_tip_updated = False
            tip_updated = False
            for i in range(0, len(miners)):
                if not (miners[i].adversarial):
                    if i == 0:
                        min_prob = miners[i].probability
                        miners[i].probability = np.random.exponential(
                            1 / ((hash_power) * lam)
                        )
                        new_node = Node(
                            parent=miners[i].mining_node,
                            weight=1,
                            advers=miners[i].adversarial,
                            children=[],
                        )

                        miners[i].mining_node.children.append(
                            new_node
                        )  # create new block
                        update_weights(new_node)
                        old_tip = tip
                        tip = update_tip(Genesis)
                        if old_tip != tip:
                            tip_updated = True
                        block_count += 1
                    else:
                        # now go through and use probability that miners moves over to new block
                        random_value = random.randint(1, 100)
                        if random_value <= p and tip_updated:
                            tip = update_tip(Genesis)
                            miners[i].mining_node = tip

                            miners[i].probability = np.random.exponential(
                                1 / ((hash_power) * lam)
                            )
                        else:
                            miners[i].probability -= min_prob
                else:  # adversarial miner, essentially same but they mine off each other
                    if i == 0:
                        min_prob = miners[i].probability
                        miners[i].probability = np.random.exponential(
                            1 / ((hash_power) * lam)
                        )
                        new_node = Node(
                            parent=miners[i].mining_node,
                            weight=1,
                            advers=miners[i].adversarial,
                            children=[],
                        )
                        miners[i].mining_node.children.append(
                            new_node
                        )  # create new block
                        update_weights(new_node)
                        old_tip = tip
                        tip = update_tip(Genesis)
                        if old_tip != tip:
                            tip_updated = True
                        block_count += 1
                        adversary_tip = new_node
                        adversary_tip_updated = True
                    else:
                        # now go through and use probability that miners moves over to new block
                        if adversary_tip_updated:
                            # all should then mine on this with certain probability
                            random_value1 = random.randint(1, 100)
                            if (
                                random_value1 <= adversary_prob
                            ):  # if less then they focus on heaviest end
                                if (
                                    adversary_tip == tip
                                ):  # if they are the same, then update
                                    miners[i].probability = np.random.exponential(
                                        1 / ((hash_power) * lam)
                                    )
                                    miners[i].mining_node = adversary_tip
                                # if the tips are not equal, then they keep mining where they are
                                else:
                                    miners[i].probability -= min_prob
                            else:  # focus on adversary chain entirely
                                miners[i].probability = np.random.exponential(
                                    1 / ((hash_power) * lam)
                                )
                                miners[i].mining_node = adversary_tip
                        else:  # normal miner was updated, so adversary will move to heaviest or stay
                            random_value2 = random.randint(1, 100)
                            if (
                                random_value2 <= adversary_prob and tip_updated
                            ):  # if less then they focus on heaviest end
                                miners[i].probability = np.random.exponential(
                                    1 / ((hash_power) * lam)
                                )
                                miners[i].mining_node = tip
                            else:
                                miners[i].probability -= min_prob

        x.append(x_val)
        tip = update_tip(Genesis)
        y.append(chain_quality(tip))

    plt.plot(x, y, label="P = " + str(p / 100.0))

    # adversary starts off not working
    # pass units of time
plt.title("Chain Quality versus Adversary hash power in GHOST")
plt.xlabel("# Adversary Nodes")
plt.ylabel("Chain Quality (%)")
plt.grid(axis = 'y')
plt.legend()
plt.show()


# update weights


# adv_prob = np.random.exponential(1 / ((beta) * lam))
# miner_prob = np.random.exponential(1 / ((1 - beta) * lam)) #lesser one typically has higher probability


# for balance can do throughput as adversarial hash power changes
# can also do number of swings per epoch as adversarial hash power changes or as adverarial partition power changes
