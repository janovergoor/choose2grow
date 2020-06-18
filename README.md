
# Discrete choice models for network formation

This code and data repository accompanies the following two papers:

1. [Choosing to grow a graph](https://arxiv.org/abs/1811.05008) - <a href="http://janovergoor.github.io/">Jan Overgoor</a>, <a href="http://www.cs.cornell.edu/~arb/">Austin R. Benson</a>, <a href="http://web.stanford.edu/~jugander/">Johan Ugander</a>. (WWW, 2019)
2. [Scaling choice models of relational social data](http://arxiv.org/abs/2006.10003) - <a href="http://janovergoor.github.io/">Jan Overgoor</a>, <a href="https://github.com/pakapol">George Pakapol Supaniratisai</a>, <a href="http://web.stanford.edu/~jugander/">Johan Ugander</a>. (KDD, 2020)

In the filenames and documentation we sometimes refer to Paper #1 and Paper #2 instead.

For questions, please email Jan at overgoor@stanford.edu.

The code for fitting logit models, as well as the code to generate the synthetic graphs, is written in Python 3. The code for the plots is written in R.

We used the following versions of external python libraries:

* `networkx=2.1`
* `numpy=1.18.1`
* `scipy=1.2.0`
* `pandas=0.23.0`
* `torch=0.4.0` (to accelerate the optimizing routine)
* `plfit` - install from [here](https://github.com/keflavich/plfit/tree/master/plfit), but remove `plfit_v1.py` before building, for Python 3 compatibility.

Instructions to reproduce the results for both papers can be found in the corresponding `README.md` files.


### Code for 'Choosing to Grow a Graph' (Paper #1)

All of the following steps are also encoded in `paper1_driver.sh`.

To reproduce the results from Section 4.1 and 4.2, follow these steps (from the `/src/paper1` folder):

1. Generate synthetic graphs with `python synth_generate.py`. This generates 10 graphs for each (r, p) combination, and writes them to `data_path/graphs`, as defined in `util.py`.
2. Extract, for each edge, the relevant choice data with `python synth_process.py`. The choice set data is written to `data_path/choices`.
3. Run the analysis code with `python make_plot_data.py`.

For the analysis in Section 4.3, follow these steps:

1. Download the Flickr data with `curl -O -4 http://socialnetworks.mpi-sws.org/data/flickr-growth.txt.gz data/`. This file is about 141 Mb large.  
2. Process the Flickr data with `python flickr_process.py`. This code takes a while to run.
3. Build the RMarkdown report with `R -e "rmarkdown::render('../paper1_reports/flickr_data.Rmd', output_file='../paper1_reports/flickr_data.pdf')"`.

For the analysis in Section 4.4, follow these steps:

1. Download the Microsoft Academic Graph. Warning, the uncompressed size of this data set is over 165Gb. Download it with the following Bash code:
    ```
    mkdir ~/mag_raw
    cd ~/mag_raw
    for i in {0..8}
    do
       curl -O -4 https://academicgraphv1wu.blob.core.windows.net/aminer/mag_papers_$i.zip
       unzip mag_papers_$i.zip
    done
    ```
2. Process the data with `python mag_process.py`. Note that you can change the field of study to process. This code takes a while to run.
2. Build the RMarkdown report with `R -e "rmarkdown::render('../paper1_reports/mag_climatology.Rmd', output_file='../paper1_reports/mag_climatology.pdf')"`.

Finally, to produce the figures of the paper, run the R code to make the plots with `Rscript make_plots.R`.


### Code for 'Scaling choice models of relational social data' (Paper #2)

The code for the simulation experiments use a _different_ implementation of the conditional logit fitting procedure than in Paper #1. This code base contains the following routine:

* Simulating graph edge formation under both regular conditional logit (single-mode multinomial) choice model and mixed mode multinomial choice model
* Feature extraction under different hyperparameters:
  - Sampling methods
  - Candidates subsampling size
  - Events subsampling size
* Choice model fitting (single and de-mixed).

To reproduce the figures, go through the following steps (from the `/src/paper2` folder). They are also encoded in `paper2_driver.sh`.

1. To generate the synthetic data for Figure 2, run `python complexity.py`
2. To generate the synthetic graph and subsequent analysis for Figure 3, run: `python synth_generate.py mnl; python synth_experiment.py fig3`.
3. To generate the synthetic graph and subsequent analysis for Figure 4, run: `python synth_generate.py mixed_mnl; python synth_experiment.py fig4`.
4. Finally, to produce the figures of the paper, run the R code to make the plots with `Rscript make_plots.R`.


### Other software libraries

Because discrete choice models are widely studied in other fields, there are many other software libraries available for the major statistical programming languages. For Python, there is an implementation in [`statsmodels`](https://www.statsmodels.org/dev/examples/notebooks/generated/discrete_choice_example.html), as well as the [`larch`](https://larch.readthedocs.io/en/latest/), [`pylogit`](https://pypi.org/project/pylogit/), [`choix`](https://github.com/lucasmaystre/choix), and [`choicemodels`](https://github.com/UDST/choicemodels) packages. For R, there are the [`mlogit`](https://cran.r-project.org/web/packages/mlogit/vignettes/mlogit.pdf) and [`mnlogit`](https://cran.r-project.org/web/packages/mnlogit/vignettes/mnlogit.pdf) libraries. Stata has the [`clogit`](https://www.stata.com/manuals13/rclogit.pdf) and [`xtmelogit`](https://www.stata.com/help11.cgi?xtmelogit) routines build-in, and there are a number of user written routes as well. We haven't tested these libraries, but they might be useful.

