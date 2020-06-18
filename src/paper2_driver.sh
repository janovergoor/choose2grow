
# time complexity analysis for Figure 2
python << END
from paper2.complexity import *
run_complexity()
END

# synthetic experiment for Figure 3
python paper2/synth_generate.py mnl
python paper2/synth_experiment.py fig3

# synthetic experiment for Figure 4
python paper2/synth_generate.py mixed-mnl
python paper2/synth_experiment.py fig4

# produce plots
Rscript paper2/make_plots.R
