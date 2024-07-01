from wrk2LoadGenerator import Wrk2LoadGenerator, Wrk2LoadGeneratorAsyn
from jaegerCollector import JaegerCollector
from k8sManager import K8sManager
from time import sleep, time
import asyncio

def interpolate_distribution(start_distribution, end_distribution, current_round, total_rounds):
    weight_start = (total_rounds - current_round) / total_rounds
    weight_end = 1 - weight_start

    interpolated_distribution = [
        weight_start * start_dist + weight_end * end_dist
        for start_dist, end_dist in zip(start_distribution, end_distribution)
    ]

    return interpolated_distribution

def reset_env(k8sManager, cpu: int = 40, mem: int = 100, replicas: int = 1):
    deployments = ["frontend", "recommendation", "search", "user", "rate", \
                   "memcached-rate", "profile", "memcached-profile", "geo", \
                    "reservation"]
    for d in deployments:
        k8sManager.set_cpu(d, cpu, cpu)
        k8sManager.set_memory(d, mem, mem)

    sleep(5)

    k8sManager.scale_deployment("frontend", replicas)
    k8sManager.scale_deployment("recommendation", replicas)
    k8sManager.scale_deployment("search", replicas)
    k8sManager.scale_deployment("user", replicas)
    k8sManager.scale_deployment("rate", replicas)
    k8sManager.scale_deployment("memcached-rate", replicas)
    k8sManager.scale_deployment("profile", replicas)
    k8sManager.scale_deployment("memcached-profile", replicas)
    k8sManager.scale_deployment("geo", replicas)
    k8sManager.scale_deployment("reservation", replicas)
    k8sManager.scale_deployment("memcached-reserve", replicas)
    k8sManager.scale_deployment("jaeger", 1)

def exp(rate: int, sub_file_name: str = None):
    # WRK2
    command_path = "/home/que/Downloads/POBO/wrk2/wrk"
    svc_url = "http://localhost:5000/"
    task_type= ""
    script = "/home/que/Downloads/POBO/wrk2/scripts/hotel-reservation/search.lua"

    task_type = "hotels"
    threads = 10
    connections = 100
    latency = True
    duration = 1500
    distribution = "exp"

    generator = Wrk2LoadGenerator(command_path, svc_url, duration, rate, threads, connections, latency, distribution, script)
    start_time = generator.generate_load()
    sleep(1)

    # JAEGER
    jaeger_url = "http://localhost:16686/api/traces"
    limit = 25000
    entry = "frontend"

    jaegerCollector = JaegerCollector(jaeger_url)
    microsvcs_latency = jaegerCollector.collect(
            start_time=start_time, duration=duration, limit=limit, service=entry, task_type=task_type, sub_file_name = sub_file_name)

    return microsvcs_latency