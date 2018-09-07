# Choosing to Grow a Graph

Research code for "Choosing to grow a graph" project. Contains code for network generation and model estimation.

The steps are as follows:

1. Generate synthetic graphs with `driver1_generate.py`. This generates 100 graphs for each (r, p) combination, and writes them to `graphs_path`, as defined in `env.py`.
2. Extract, for each edge, the relevant network context with `driver2_process.py`.
3. Fit multinomial logit models with `driver3_estimate.py`.

