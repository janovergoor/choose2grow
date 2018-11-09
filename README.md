# Choosing to Grow a Graph

This code and data repository accompanies the paper

- [Choosing to grow a graph](..). Jan Overgoor, Austin R. Benson, Johan Ugander. 2018.

For questions, please email Jan at overgoor@stanford.edu.

The code for fitting logit models, as well as the code to generate the synthetic graphs for section 4.1, is written in Python 3. The code for the plots is written in R.


### Reproducing results and figures

To reproduce the results from Section 4.1 and 4.2, follow these steps (from the `/src` folder):

1. Generate synthetic graphs with `python synth_generate.py`. This generates 10 graphs for each (r, p) combination, and writes them to `data_path/graphs`, as defined in `util.py`.
2. Extract, for each edge, the relevant choice data with `python synth_process.py`. The choice set data is written to `data_path/choices`.
3. Run the analysis code with `python make_plot_data.py`.

For the analysis in Section 4.3, follow these steps:

1. Download the Flickr data with `wget -4 http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz ../data/`.
2. Process the Flickr data with `python flickr_process.py`.
3. Build the RMarkdown report with `R -e "rmarkdown::render('../reports/flicrk_data.Rmd', output_file='../reports/flicrk_data.pdf')"`.

For the analysis in Section 4.4, follow these steps:

1. Download the Microsoft Academic Graph data with the following Bash code:
    ```
    mkdir ~/mag_raw
    cd mag_raw
    for i in {0..8}
    do
       wget -4 https://academicgraphv1wu.blob.core.windows.net/aminer/mag_papers_$i.zip
       unzip mag_papers_$i.zip
    done
    ```
2. Process the data with `python mag_process.py`. Note that you can change the field of study to process.
2. Build the RMarkdown report with `R -e "rmarkdown::render('../reports/mag_climatology.Rmd', output_file='../reports/mag_climatology.pdf')"`.

Finally, to produce the figures of the paper, run the R code to make the plots with `Rscript make_plots.R`.


### Other software libraries

Because discrete choice models are widely studied in other fields, there are many other software libraries available for the major statistical programming languages. For Python, there is an implementation in [`statsmodels`](https://www.statsmodels.org/dev/examples/notebooks/generated/discrete_choice_example.html), as well as the [`larch`](https://larch.readthedocs.io/en/latest/), [`pylogit`](https://pypi.org/project/pylogit/), and [`choicemodels`](https://github.com/UDST/choicemodels) packages. For R, there are the [`mlogit`](https://cran.r-project.org/web/packages/mlogit/vignettes/mlogit.pdf) and [`mnlogit`](https://cran.r-project.org/web/packages/mnlogit/vignettes/mnlogit.pdf) libraries. Stata has the [`clogit`](https://www.stata.com/manuals13/rclogit.pdf) and [`xtmelogit`](https://www.stata.com/help11.cgi?xtmelogit) routines build-in, and there are a number of user written routes as well. We haven't tested these libraries, but they might be useful.


