# Lab roadmap

## Seed milestone

- [x] Project skeleton
- [x] Cross-project remit document
- [x] Cascade-Correlation XOR lab
- [x] Metadata validator
- [x] Minimal tests

## Near-term candidates

1. Perceptron and ADALINE basics
   - [x] deterministic perceptron trace on a tiny linearly separable dataset
   - [ ] ADALINE comparison
   - [ ] learning-rate and margin visualisation

2. Dynamic Node Construction toy example
   - add hidden nodes when residual error stalls
   - compare against fixed hidden-layer baseline

3. Growing Neural Gas mini-demo
   - 2D point cloud adaptation
   - node insertion trace

4. STDP timing-window learner
   - synaptic selectivity toy trace
   - explicitly not neuron-growth unless linked to reviewed structural-growth sources

5. Post-growth stabilization comparison
   - freeze vs fine-tune vs prune after adding capacity

## Project health goals

- Every `available` lab has an `experiment.json`, README, runnable script or static demo, and validation coverage.
- Daily automation improves one small artifact and commits locally, but never pushes.
- The lab links to the literature garden rather than duplicating its paper reviews.
