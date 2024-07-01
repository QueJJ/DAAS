from algo import DAAS
from k8sManager import K8sManager
import numpy as np
from time import sleep
from utils import reset_env, exp
import json
import networkx as nx
import os
import csv
import pandas as pd
import time
import subprocess

module_name = "search"
sla = 10

k8sManager = K8sManager("hotel-reserv")

repeats = 1
periods = 300
rounds = 10
rates = 80
duration = '30s'

# DAG graph info
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3, 4, 5])
G.add_edges_from([(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)])
topological_order = list(nx.topological_sort(G))
paths = []
sources = [0]
targets = [1, 2, 4, 5]
for s in sources:
    for t in targets:
        paths.extend(list(nx.all_simple_paths(G, source=s, target=t)))
contexts = [np.arange(30, -1, -3), np.arange(30, -1, -3), np.arange(60, -1, -3),
            np.arange(30, -1, -3), np.arange(30, -1, -3), np.arange(30, -1, -3)]

# set the upper bound of pods for each service
UB = 8
px = range(1, UB)
py = range(1, UB)
pz = range(1, UB)
pa = range(1, UB)
pb = range(1, UB)
pc = range(1, UB)

# main body
def run():
    avg_reward = np.zeros(periods)
    avg_cost = np.zeros(periods)
    for j in range(repeats):
        reset_env(k8sManager)
        model = DAAS(np.meshgrid(px, py, pz, pa, pb, pc), contexts, G, paths)

        reward_sum = []
        cost_sum = []
        reward_list = []
        latency_list = []

        # save data
        folder_name = f"NewRes/{module_name}_{alg}_{j+1}/"
        file_name = folder_name + f"{module_name}.csv"
        os.system("mkdir "+folder_name)
        with open(file_name, 'w', newline="") as f:
            f.truncate()
            csv.writer(f).writerow(["resource_usage","cp_latency"])

        for i in range(periods):
            print("="*90)
            # Make decision
            pod_0, pod_1, pod_2, pod_3, pod_4, pod_5 = model.decision()
            print(f"pod num: {pod_0,pod_1,pod_2,pod_3,pod_4,pod_5}")

            # Execute action
            k8sManager.scale_deployment("frontend", int(pod_0))
            k8sManager.scale_deployment("profile", int(pod_1))
            k8sManager.scale_deployment("reservation", int(pod_2))
            k8sManager.scale_deployment("search", int(pod_3))
            k8sManager.scale_deployment("geo", int(pod_4))
            k8sManager.scale_deployment("rate", int(pod_5))

            sleep(2)
            # Repeat to get average latency, better experiment result
            ml_sum = [[], [], [], [], [], []]
            for _ in range(rounds):
                sub_file_name = f"{folder_name}/{i}.csv"
                with open(sub_file_name, 'w', newline="") as f:
                    f.truncate()
                    csv.writer(f).writerow(["service-name","latency"])
                while True:
                    ml = exp(rates, sub_file_name)
                    if ml==[]:
                        time.sleep(2)
                        continue
                    for q in range(len(ml)):
                        ml_sum[q].append(ml[q])
                    sleep(5)
                    break

            # Update model
            cost = []
            for p in range(len(ml_sum)):
                cost.append(sum(ml_sum[p]) / len(ml_sum[p]) / 1000)
            segments_latency = np.zeros(len(cost))
            for node in topological_order:
                parents = list(G.predecessors(node))
                if parents != []:
                    segments_latency[node] = cost[node] + max([segments_latency[p] for p in parents])
                else:
                    segments_latency[node] = cost[node]
            
            reward = 42 - np.sum([pod_0,pod_1,pod_2,pod_3,pod_4,pod_5])
            el = max(segments_latency)
            # scale the segments latency for better learning
            scaled_segments_latency= segments_latency / 1000
            model.update(reward, scaled_segments_latency)

            reward_list.append(reward)
            latency_list.append(el)

            reward_sum.append(reward + reward_sum[-1] if reward_sum else reward)
            cost_sum.append(el + cost_sum[-1] if cost_sum else el)

            print(f"Period {i+1}: Pod Num.: {np.sum(reward)}, P90 TL: {el}")
            with open(file_name, 'a', newline="") as f:
                csv.writer(f).writerow([str(reward),el])

        reward_sum = [reward_sum[i]/(i+1) for i in range(periods)]
        cost_sum = [cost_sum[i]/(i+1) for i in range(periods)]

        avg_reward += np.array(reward_sum)
        avg_cost += np.array(cost_sum)

    avg_reward /= repeats
    avg_cost /= repeats

run()