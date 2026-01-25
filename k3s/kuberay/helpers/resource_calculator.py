"""Calculadora de recursos para anticipar cuántos CPUs necesita Tune+Train."""

from __future__ import annotations

import argparse
import os
import sys
from math import ceil
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from resources import ResourceConfig


def max_concurrent_trials(total_cluster_cpus: int, cfg: ResourceConfig) -> int:
    per_trial = cfg.total_cpus_per_trial()
    if per_trial <= 0:
        raise ValueError("per trial CPU must be positive")
    return total_cluster_cpus // per_trial


def required_cluster_cpus(target_trials: int, cfg: ResourceConfig) -> int:
    return cfg.total_cpus_per_trial() * target_trials


def print_summary(total_cluster_cpus: int, cfg: ResourceConfig) -> None:
    print("\nResumen de recursos calculados:")
    print(f"  Workers por trial: {cfg.num_workers}")
    print(f"  CPUs por worker: {cfg.cpus_per_worker}")
    print(f"  CPUs destinados al head: {cfg.cpus_head}")
    print(f"  Total por trial (head + workers): {cfg.total_cpus_per_trial()}")
    max_trials = max_concurrent_trials(total_cluster_cpus, cfg)
    print(f"  Máximo de trials concurrentes con {total_cluster_cpus} CPUs: {max_trials} (teórico)")


def print_node_assessment(
    cluster_nodes: Optional[int], cpus_per_node: Optional[int], cfg: ResourceConfig, desired_trials: Optional[int]
) -> None:
    if not cluster_nodes or not cpus_per_node:
        return

    print("\nVerificación de arquitectura de nodos (Bin Packing):")
    total_node_cpus = cluster_nodes * cpus_per_node
    print(f"  Configuración: {cluster_nodes} nodos × {cpus_per_node} CPUs = {total_node_cpus} CPUs totales")

    # ¿Cabe un worker en un nodo?
    if cpus_per_node < cfg.cpus_per_worker:
        print(f"  [ERROR] INFACTIBLE: El worker pide {cfg.cpus_per_worker} CPUs y el nodo solo tiene {cpus_per_node}.")
        return

    # ¿Cuántos workers caben por nodo?
    workers_per_node = cpus_per_node // cfg.cpus_per_worker
    waste_per_node = cpus_per_node % cfg.cpus_per_worker
    print(f"  Capacidad real: {workers_per_node} worker(s) por nodo.")
    if waste_per_node > 0:
        print(f"  Aviso: Cada nodo desperdicia {waste_per_node} CPU(s) debido al tamaño del worker.")

    # ¿Cuántos nodos consume un trial?
    nodes_for_workers = ceil(cfg.num_workers / workers_per_node)
    print(f"  Un solo trial consumirá aproximadamente {nodes_for_workers} nodo(s) para sus workers.")

    if desired_trials:
        # Recomendar replicas para el YAML
        print("\nRecomendación para autoscaling (RayJob YAML):")
        # minReplicas: al menos lo necesario para 1 trial
        min_reps = nodes_for_workers
        # maxReplicas: lo necesario para los trials deseados
        max_reps = nodes_for_workers * desired_trials
        
        print(f"  minReplicas recomendadas: {min_reps} (para asegurar que 1 trial pueda arrancar siempre)")
        print(f"  maxReplicas recomendadas: {max_reps} (para permitir {desired_trials} trials paralelos)")

        # CPUs teóricos vs reales
        needed_cpus = required_cluster_cpus(desired_trials, cfg)
        if total_node_cpus >= needed_cpus:
            print(f"\n  [OK] CPUs suficientes: Tienes {total_node_cpus}, necesitas {needed_cpus}.")
        else:
            print(f"\n  [CRÍTICO] CPUs insuficientes: Faltan {needed_cpus - total_node_cpus} CPUs.")

        # Nodos necesarios considerando fragmentación
        total_nodes_needed = ceil(needed_cpus / cpus_per_node)
        if cluster_nodes < total_nodes_needed:
            print(f"  [!] Alerta: El clúster físico es pequeño. Necesitas que el Cluster Autoscaler de K8s pueda subir hasta {total_nodes_needed} nodos.")


if __name__ == "__main__":
    DEFAULT_CLUSTER_CPUS = 32
    parser = argparse.ArgumentParser(
        description="Calcula cuánto CPU necesitas y recomienda parámetros de autoscaling."
    )
    parser.add_argument("--cluster-cpus", type=int, default=DEFAULT_CLUSTER_CPUS, help="CPUs totales del clúster")
    parser.add_argument("--cluster-nodes", type=int, help="Número de nodos físicos")
    parser.add_argument("--cpus-per-node", type=int, help="CPUs por cada nodo")
    parser.add_argument("--desired-trials", type=int, help="Trials paralelos objetivo")
    parser.add_argument("--num-workers", type=int, help="Workers de Ray Train por trial")
    parser.add_argument("--cpus-per-worker", type=int, help="CPUs por cada worker")
    parser.add_argument("--cpus-head", type=int, help="CPU para el driver del trial")
    args = parser.parse_args()

    env_cfg = ResourceConfig.from_env()
    cfg = ResourceConfig(
        num_workers=args.num_workers if args.num_workers is not None else env_cfg.num_workers,
        cpus_per_worker=args.cpus_per_worker if args.cpus_per_worker is not None else env_cfg.cpus_per_worker,
        cpus_head=args.cpus_head if args.cpus_head is not None else env_cfg.cpus_head,
    )

    cluster_cpus = args.cluster_cpus
    if args.cluster_nodes and args.cpus_per_node:
        if args.cluster_cpus == DEFAULT_CLUSTER_CPUS:
            cluster_cpus = args.cluster_nodes * args.cpus_per_node

    print_summary(cluster_cpus, cfg)
    print_node_assessment(args.cluster_nodes, args.cpus_per_node, cfg, args.desired_trials)

    if args.desired_trials:
        needed = required_cluster_cpus(args.desired_trials, cfg)
        print(f"\nResultado final: Necesitas {needed} CPUs para {args.desired_trials} trials.")
