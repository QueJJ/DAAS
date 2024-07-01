# DAAS: Dependency-aware Auto-scaling for Efficient Microservice Workflow Orchestration

## Setup environment
* A Kubernetes cluster with the microservice system in [DeathstarBench](https://github.com/delimitrou/DeathStarBench)
* Set up the Python environment in [environment.yml](https://github.com/QueJJ/DAAS/edit/main/environment.yml)
* Install [Prometheus](https://prometheus.io/) on your Kubernetes cluster to observe Jaeger's overhead.

## Experiments
Run the following command to see the performance of DAAS:

`python run.py`
