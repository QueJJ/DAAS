# DAAS: Dependency-aware Auto-scaling for Efficient Microservice Workflow Orchestration

## Setup environment
* A Kubernetes cluster with the microservice system in [DeathstarBench](https://github.com/delimitrou/DeathStarBench)
* Set up the Python environment in [environment.yml](https://github.com/QueJJ/DAAS/edit/main/environment.yml)
* Install [Prometheus](https://prometheus.io/) on your Kubernetes cluster to observe Jaeger's overhead.

## Modules
We implement different modules of DAAS in separate Python scripts as follows:

* Algorithm: Main algorithm of DAAS, including Learning and Decision module.
* Jaeger Collector: We use Jaeger to trace the latency of the microservice application. DAAS collects per-service tracing data through Jaeger's exposed RESTful APIs
* Resource Manager: We implement the pod number controller by leveraging the Kubernetes Python client.
* Workload Generator: We use the commonly used HTTP benchmarking tool wrk2 as the workload generator to send requests. Wrk2 provides APIs for setting different thread numbers, the number of HTTP connections, the number of requests, the duration, and etc. You can change these parameters according to your cluster configuration.

## Experiments
### Results in paper
Run the following command to see the performance of DAAS:

`python run.py`

### Customized microservice application
To run your customized microservice application, you need to modify the **DAG graph info** in the file *run.py* to fit to your application's DAG.

## Run Derm Algorithm
Modify the rate, microservice name in the Derm, and select different sample allocations to train models, and select the best model.
git: https://github.com/liaochan/Derm
