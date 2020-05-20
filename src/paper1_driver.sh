
# generate synthetic graphs
python << END
from paper1.synth_generate import *
run_generate()
from paper1.synth_process import *
run_process()
END

# make data for plots 1 to 4
python << END
from paper1.make_plot_data import *
make_figure1_data()
make_figure2_data()
make_figure3_data()
make_figure4_data()
END

# process Flickr data
curl -O -4 http://socialnetworks.mpi-sws.org/data/flickr-growth.txt.gz ../data/
python paper1/flickr_process.py
R -e "rmarkdown::render('paper1_reports/flickr_data.Rmd', output_file='paper1_reports/flickr_data.pdf')"

# process Microsoft Academic Graph data
mkdir ~/mag_raw
cd ~/mag_raw
for i in {0..8}
do
   curl -O -4 https://academicgraphv1wu.blob.core.windows.net/aminer/mag_papers_$i.zip
   unzip mag_papers_$i.zip
done
python paper1/mag_process.py
R -e "rmarkdown::render('paper1_reports/mag_climatology.Rmd', output_file='paper1_reports/mag_climatology.pdf')"

# produce plots
Rscript paper1/make_plots.R