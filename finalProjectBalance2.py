import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import random

# in context of ethereum

# partitioning power of adversary
# assume that only 2 partitions exist - more can exist
# with partitioning, paritioned parties cannot see each other
# adversaries are not mining in this because this is a function of partitioning power
# now have modification where adversaries can also add blocks of their own


class Node:
    def __init__(self, parent, children, weight, partition):
        self.parent = parent
        self.children = children
        self.weight = weight
        self.partition = partition


class Worker:
    def __init__(self, partition, probability, mining_node):
        self.partition = partition  # either 0 or 1 or 2 (bad)
        self.probability = probability
        self.mining_node = mining_node


Genesis0 = Node(parent=None, children=[], weight=1, partition=0)
Genesis1 = Node(parent=None, children=[], weight=1, partition=1)
def get_chain(node):
    chain_list = []
    curr_node = node
    while True:
        if (curr_node.parent == None):
            return chain_list
        chain_list = chain_list + [curr_node]
        curr_node = curr_node.parent
        
def good_length(list):
    count = 0
    for i in range(len(list)):
        if list[i].partition == 0 or list[i].partition == 1:
            count += 1
    return count
      
# 0 indicates node1 heavier, 1 if node 2 heavier, 2 if same
def heaviest_chain(node1, node2):
    
    node1_chain = get_chain(node1)
    node2_chain = get_chain(node2)

    length = min([len(node1_chain), len(node2_chain)])
    for i in range(length):
        
        if (node2_chain[i].weight < node1_chain[i].weight): #node 1 is heavier
            0
        else: 
            if (node1_chain[i].weight < node2_chain[i].weight):
                1
    if (len(node2_chain) < len(node1_chain)):
        return 0
    else: 
        if (len(node1_chain) < len(node2_chain)):
          return 1
        
    
    return 2


def update_weights(node):
    curr_node = node
    while True:
        if curr_node.parent == None:
            break
        curr_node.parent.weight += 1
        curr_node = curr_node.parent


def update_tip(node):  # fix this
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
        if curr_node.advers:
            total_count += 1.0
        else:
            total_count += 1.0
            good_count += 1.0
        curr_node = curr_node.parent
    
    return good_count / total_count


lam = 1
total_and_bad = 50
adversary_power = 20
total_miners = (
    total_and_bad - adversary_power
)  # 20 nodes equivalent to 20 percent power


for p in [50,100]:  # probability that miner changes over to new block
    x = []
    y = []
    for partition_power in range(0, 101):
        x_val = partition_power
        miners = []

        Genesis0 = Node(parent=None, children=[], weight=1, partition=0)
        Genesis1 = Node(parent=None, children=[], weight=1, partition=1)

        block_count = 1
        true_tip = 0  # 0 means tip belongs to Genesis0
        partition0_tip = Genesis0
        partition1_tip = Genesis1
        hash_power_good = 1 / total_and_bad
        switches = 0
        # create miners in different partitions
        for i in range(0, total_and_bad):
            if i < (partition_power / 100) * total_miners:
                miners.append(
                    Worker(partition=0, probability=0.0, mining_node=Genesis0)
                )
            else:
                if i < (total_and_bad - adversary_power):
                    miners.append(
                        Worker(partition=1, probability=0.0, mining_node=Genesis1)
                    )
                else:
                    miners.append(
                        Worker(partition=2, probability=0.0, mining_node=Genesis0)
                    )

            miners[i].probability = np.random.exponential(1 / ((total_and_bad) * lam))

        while block_count < 500:
            miners.sort(key=lambda x: x.probability)
            min_prob = 0
            tip0_updated = False
            tip_updated = False
            tip1_updated = False
            for i in range(0, len(miners)):
                if miners[i].partition == 0:
                    if i == 0:
                        min_prob = miners[i].probability
                        miners[i].probability = np.random.exponential(
                            1 / ((total_and_bad) * lam)
                        )
                        new_node = Node(
                            parent=miners[i].mining_node,
                            weight=1,
                            partition=0,
                            children=[],
                        )

                        miners[i].mining_node.children.append(
                            new_node
                        )  # create new block
                        update_weights(new_node)
                        old0_tip = partition0_tip
                        partition0_tip = update_tip(Genesis0)
                        if old0_tip != partition0_tip:
                            
                            tip0_updated = True
                            if (
                                (heaviest_chain(partition0_tip, partition1_tip) == 0) and true_tip == 1
                            ) or ((heaviest_chain(partition0_tip, partition1_tip) == 2)):
                                
                                true_tip = 0
                                switches += 1
                            else:
                                if (heaviest_chain(partition0_tip, partition1_tip) == 0):
                                    true_tip = 0

                        block_count += 1
                    else:
                        # now go through and use probability that miners moves over to new block
                        random_value = random.randint(1, 100)
                        if tip0_updated and random_value <= p:
                            partition0_tip = update_tip(Genesis0)
                            miners[i].mining_node = partition0_tip

                            miners[i].probability = np.random.exponential(
                                1 / ((total_and_bad) * lam)
                            )
                        else:
                            miners[i].probability -= min_prob
                else:  # miner in other partition, essentially same but they mine off each other
                    if miners[i].partition == 1:
                        if i == 0:
                            min_prob = miners[i].probability
                            miners[i].probability = np.random.exponential(
                                1 / ((total_and_bad) * lam)
                            )
                            new_node = Node(
                                parent=miners[i].mining_node,
                                weight=1,
                                partition=1,
                                children=[],
                            )

                            miners[i].mining_node.children.append(
                                new_node
                            )  # create new block
                            update_weights(new_node)
                            old1_tip = partition1_tip
                            partition1_tip = update_tip(Genesis1)
                            if old1_tip != partition1_tip:
                                
                                tip1_updated = True
                                if (
                                    (heaviest_chain(partition0_tip, partition1_tip) == 1) and true_tip == 0
                                ) or (heaviest_chain(partition0_tip, partition1_tip) == 2):
                                    
                                      
                                    true_tip = 1
                                    switches += 1
                                else:
                                    if (heaviest_chain(partition0_tip, partition1_tip) == 1):
                                        true_tip = 1
                            block_count += 1
                        else:
                            # now go through and use probability that miners moves over to new block
                            random_value = random.randint(1, 100)
                            if random_value <= p and tip1_updated:
                                partition1_tip = update_tip(Genesis1)
                                miners[i].mining_node = partition1_tip

                                miners[i].probability = np.random.exponential(
                                    1 / ((total_and_bad) * lam)
                                )
                            else:
                                miners[i].probability -= min_prob
                    else: # adversaries 
                        if i ==0: 
     
                            block_count +=1 
                            min_prob = miners[i].probability
                   
                            miners[i].probability = np.random.exponential(
                                1 / ((total_and_bad) * lam)
                            )
                    
                            if heaviest_chain(partition0_tip, partition1_tip) == 0:  # add block to Genesis 1
                                new_node = Node(
                                parent=partition1_tip,
                                weight=1,
                                partition=2,
                                children=[],
                                )

                                partition1_tip.children.append(
                                    new_node
                                )  # create new block
                                update_weights(new_node)
                                old1_tip = partition1_tip
                                partition1_tip = update_tip(Genesis1)
                                if old1_tip != partition1_tip:
                                      tip1_updated = True
                                      if (
                                          (heaviest_chain(partition0_tip, partition1_tip) == 1) and true_tip == 0
                                      ) or (heaviest_chain(partition0_tip, partition1_tip) == 2):
                                          true_tip = 1
                                          switches += 1
                                      else:
                                          if (heaviest_chain(partition0_tip, partition1_tip) == 1):
                                              true_tip = 1
                            
                            else:
                                if heaviest_chain(partition0_tip, partition1_tip) == 1: # add block to genesis 0
                                    
                                    new_node = Node(
                                        parent=partition0_tip,
                                        weight=1,
                                        partition=2,
                                        children=[],
                                    )

                                    partition0_tip.children.append(
                                        new_node
                                    )  # create new block
                                    update_weights(new_node)
                                    old0_tip = partition0_tip
                                    partition0_tip = update_tip(Genesis0)
                                    if old0_tip != partition0_tip:
                                        tip0_updated = True
                                        if (
                                            (heaviest_chain(partition0_tip, partition1_tip) == 0) and true_tip == 1
                                        ) or ((heaviest_chain(partition0_tip, partition1_tip) == 2)):
                                            true_tip = 0
                                            switches += 1
                                        else:
                                            if (heaviest_chain(partition0_tip, partition1_tip) == 0):
                                                true_tip = 0
                                else: # they are the same so adversary chooses one to cause switch
                                    
                                    if (true_tip == 0): # put in chain 1
                                        new_node = Node(
                                        parent=partition1_tip,
                                        weight=1,
                                        partition=2,
                                        children=[],
                                        )

                                        partition1_tip.children.append(
                                            new_node
                                        )  # create new block
                                        update_weights(new_node)
                                        old1_tip = partition1_tip
                                        partition1_tip = update_tip(Genesis1)
                                        if old1_tip != partition1_tip:
                                              tip1_updated = True
                                              if (
                                                  (heaviest_chain(partition0_tip, partition1_tip) == 1) and true_tip == 0
                                              ) or (heaviest_chain(partition0_tip, partition1_tip) == 2):
                                                  true_tip = 1
                                                  switches += 1
                                              else:
                                                  if (heaviest_chain(partition0_tip, partition1_tip) == 1):
                                                      true_tip = 1
                                    else: # need to put in chain 0
                                        # then need to make sure that adversary is resetting every time that it does not win and the tip is updated
                                        new_node = Node(
                                        parent=partition0_tip,
                                        weight=1,
                                        partition=2,
                                        children=[],
                                        )
                                        partition0_tip.children.append(
                                            new_node
                                        )  # create new block
                                        update_weights(new_node)
                                        old0_tip = partition0_tip
                                        partition0_tip = update_tip(Genesis0)
                                        if old0_tip != partition0_tip:
                                            tip0_updated = True
                                            if (
                                                (heaviest_chain(partition0_tip, partition1_tip) == 0) and true_tip == 1
                                            ) or ((heaviest_chain(partition0_tip, partition1_tip) == 2)):
                                                true_tip = 0
                                                switches += 1
                                            else:
                                                if (heaviest_chain(partition0_tip, partition1_tip) == 0):
                                                    true_tip = 0

                        else: # adversary but they don't win, they must mine on smaller tip
                            
                            if (heaviest_chain(partition0_tip, partition1_tip) == 1 or (heaviest_chain(partition0_tip, partition1_tip) == 2)): # also only change if they were updated
                                if (tip0_updated):
                                    miners[i].probability = np.random.exponential(1 / ((total_and_bad) * lam))
                                    miners[i].mining_node = partition0_tip
                                else: 
                                    miners[i].probability -= min_prob
                                    
                                
                            else: # partition0 is denser 
                              if (tip1_updated):
                                   miners[i].probability = np.random.exponential(1 / ((total_and_bad) * lam))
                                   miners[i].mining_node = partition1_tip
                              else: 
                                    miners[i].probability -= min_prob
                                  
                            
                             
                            
                                        
                                        
                                    
                                  
                                    

                              
                                
                        

        x.append(x_val)
        
        y_val = 0
        if (heaviest_chain(partition0_tip, partition1_tip) == 0):
            y_val = good_length(get_chain(partition0_tip))/block_count
        else:
            y_val = good_length(get_chain(partition1_tip))/block_count
        print(y_val)
        y.append(y_val)

    plt.plot(x, y, label="P = " + str(p / 100.0))
    '''coeff = np.polyfit(x,y,3)
    for index in range(len(x)): 
        y[index] = coeff[0]*(x[index]**3) + coeff[1]*(x[index]**2) + coeff[2] * x[index] + coeff[3]
    plt.plot(x, y)'''

    # adversary starts off not working
    # pass units of time
plt.title("Throughput versus Adversary Partition power in GHOST")
plt.xlabel("Adversary Partition Power (%)")
plt.ylabel("Proportion of Blocks in Heaviest Chain")
plt.grid(axis="y")
plt.legend()
plt.show()


# update weights


# adv_prob = np.random.exponential(1 / ((beta) * lam))
# miner_prob = np.random.exponential(1 / ((1 - beta) * lam)) #lesser one typically has higher probability


# for balance can do throughput as adversarial hash power changes
# can also do number of swings per epoch as adversarial hash power changes or as adverarial partition power changes
