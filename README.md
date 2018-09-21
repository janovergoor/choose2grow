# Choosing to Grow a Graph

This code and data repository accompanies the paper

- [Choosing to grow a graph](..). Jan Overgoor, Austin R. Benson, Johan Ugander. 2018.

For questions, please email Jan at overgoor@stanford.edu.

The code for fitting logit models, as well as the code to generate the synthetic graphs for section 4.1, are both written in Python 3. The code for creating the graphs is written in R.

To reproduce the results from the paper, follow these steps (from the `/src` folder):

1. Generate synthetic graphs with `python generate.py`. This generates 10 graphs for each (r, p) combination, and writes them to `graphs_path`, as defined in `env.py`.
2. Extract, for each edge, the relevant choice data with `python process.py`. The choice set data is written to `data_path` as defined in `env.py`.
3. Run the analysis code with `python analyze.py > ../results/r_vs_p_synth.csv`.
4. Run the plot code with `Rscript make_plots.R`.
