# Synthetic Data And Drift

## Status
- `working`: synthetic generator with explicit drift modes
- `partial`: monitoring/retraining loop integration is not production-complete

## Design
- Synthetic traffic generator supports:
  - `normal`
  - `data_drift` (feature distribution shift)
  - `concept_drift` (relationship shift affecting target mapping)
- Kafka producer can continuously emit generated events for streaming experiments.

## Monitoring Note
A monitoring prototype exists (`evidentlyIA.py`), but it is not currently integrated as a hardened platform service in this repo.

## Trade-Offs
- Synthetic data accelerates repeatable testing but does not replace real distribution behavior.
- Drift simulation is useful for experimentation, not a substitute for production monitoring policy.

## Evidence Pointers
- `producer/syntheticgenerator.py`
- `producer/producer.py`
- `evidentlyIA.py`
