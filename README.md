# Choosing to Grow a Graph

This code and data repository accompanies the paper

- [Choosing to grow a graph](..). Jan Overgoor, Austin R. Benson, Johan Ugander. 2018.

For questions, please email Jan at overgoor@stanford.edu.

The code for fitting logit models, as well as the code to generate the synthetic graphs for section 5.1, is written in Python 3. The code for creating the graphs and analysis is written in R.

To reproduce the results from Section 5.1, follow these steps (from the `/src` folder):

1. Generate synthetic graphs with `python synth_generate.py`. This generates 10 graphs for each (r, p) combination, and writes them to `data_path/graphs`, as defined in `util.py`.
2. Extract, for each edge, the relevant choice data with `python synth_process.py`. The choice set data is written to `data_path/choices`.
3. Run the estimation code with `python synth_fit.py > ../results/r_vs_p_synth.csv`.
4. Run the R code to make the plots with `Rscript make_plots.R`.

For the results from Section 5.2, follow these steps:

1. Download the Flickr data with `wget -4 http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz ./data/`.
2. Process the Flickr data with `python flickr_process.py`.
3. Build the RMarkdown report with `R -e "rmarkdown::render('reports/flicrk_data.Rmd', output_file='reports/flicrk_data.pdf')"`.

For the results from Section 5.3, follow these steps:

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
2. Process the data with `python mag_process.py`.
2. Build the RMarkdown report with `R -e "rmarkdown::render('reports/mag_climatology.Rmd', output_file='reports/mag_climatology.pdf')"`.
