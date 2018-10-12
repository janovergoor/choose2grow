# Run with Rscript make_plots.R

library(dplyr)
library(ggplot2)
library(latex2exp)
library(stringr)
library(parallel)
library(tidyr)
library(readr)
library(scales)

setwd("~/projects/choosing_to_grow/choose2grow/src")


my_theme <- function(base_size=9) {
  # Set the base size
  theme_bw(base_size=base_size) +
    theme(
      # Center title
      plot.title = element_text(hjust = 0.5),
      # Make the background white
      panel.background=element_rect(fill='white', colour='white'),
      panel.grid.major=element_blank(),
      panel.grid.minor=element_blank(),
      # Minimize margins
      plot.margin=unit(c(0.2, 0.2, 0.2, 0.2), "cm"),
      panel.margin=unit(0.25, "lines"),
      # Tiny space between axis labels and tick labels
      axis.title.x=element_text(margin=ggplot2::margin(t=6.0)),
      axis.title.y=element_text(margin=ggplot2::margin(r=6.0)),
      # Simplify the legend
      legend.key=element_blank(),
      #legend.title=element_blank(),
      legend.title.align=0.5,
      legend.background=element_rect(fill='transparent')
    )
}


##
## Figure 1 - estimates of p by PA and PA-FoF models
##
DF <- read_csv("../results/r_vs_p_synth.csv", col_types='ccdd') %>%
      separate(fn, into=c("type","r","p","id"), sep='-') %>%
      group_by(type, r, p, model) %>%
      summarize(LL=-1*mean(ll), mean=mean(estimate),
                ll=min(estimate), ul=max(estimate),
                n=n()) %>%
      ungroup() %>%
      mutate(
        Model=model,
        Type=ifelse(type=='g', 'External / Growth', 'Internal / Densify'),
        r=as.numeric(r),
        P=paste0('p=', p)
      )

DF %>%
  filter(type=='g') %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, mean, color=p)) +
    geom_line() + #geom_point() +
    geom_segment(aes(x=r, xend=r, y=ll, yend=ul)) +
    facet_grid( ~ Model) +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of $p$"), expand=c(0,0), limits=c(0, 1.02), breaks=seq(0, 1, 0.25)) +
    my_theme(11) -> p
ggsave('../results/fig_1.pdf', p, width=6, height=3)

DF %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, mean, color=p)) +
    geom_line() + geom_point() +
    geom_segment(aes(x=r, xend=r, y=ll, yend=ul)) +
    facet_grid(Type ~ Model) +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of $p$"), expand=c(0,0), limits=c(0, 1.02), breaks=seq(0, 1, 0.25)) +
    my_theme(11) -> p
ggsave('../results/fig_1a.pdf', p, width=6, height=4)

## Figure 1b - LL by PA and PA-FoF models
## currently not in the paper
DF %>%
  filter(Model != 'p-single') %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, LL, color=Model)) +
    geom_line() +
    facet_grid(Type ~ P, scales='free_y') +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous("Log-likelihood") +
    my_theme(11) -> p
ggsave('../results/fig_1b.pdf', p, width=9, height=3.5)



##
## Figure 2 - Power law fits on degree of (r,p) graphs
##
source("http://tuvalu.santafe.edu/~aaronc/powerlaws/plfit.r")

DF <- list.files("../data/synth_graphs", pattern='[gd].*') %>%
  mclapply(function(fn){
    el <- read_csv(paste0("../data/synth_graphs/", fn), col_types='iii')
    n_nodes <- max(el$from)
    el <- el %>% group_by(to) %>% summarize(deg=n())
    times = max(n_nodes-length(el$deg), 0)
    degs = c(el$deg, rep(0,  times))
    # compute powerlaw fit
    fit <- plfit(el$deg, "range", seq(1.001,5,0.01))
    # compute jacksons r
    cdf <- el %>% group_by(deg) %>% summarize(n=n()) %>% ungroup() %>%
      arrange(deg) %>% complete(deg=seq(max(deg)), fill=list(n=0)) %>%
      mutate(F_d=cumsum(n)/sum(n))
    rstar = r_jackson(cdf$F_d, 4)
    r_est = rstar/(1+rstar)
    # gather results
    vals <- str_split(fn, '-')[[1]]
    data.frame(type=vals[1], r=vals[2], p=vals[3], id=substr(vals[4], 1, 2),
               alpha=fit$alpha, xmin=fit$xmin, r_est=r_est)
  }, mc.cores=10) %>%
  bind_rows() %>%
  group_by(type, r, p) %>%
  summarize(mean_a=mean(alpha), ll_a=quantile(alpha, 0.25), ul_a=quantile(alpha, 0.75), # ll_a=min(alpha), ul_a=max(alpha),
            mean_r=mean(r_est), ll_r=quantile(r_est, 0.25), ul_r=quantile(r_est, 0.75)
            ) %>% ungroup() %>%
  mutate(r=as.numeric(r)) %>%
  # offset r slightly
  mutate(r_off = r + (as.numeric(p)-0.5)/50)

DF %>% filter(type == 'g') %>%
ggplot(aes(x=r_off, y=mean_a, color=p)) + geom_line() +
  geom_segment(aes(x=r_off, xend=r_off, y=ll_a, yend=ul_a)) +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $\\gamma$"), expand=c(0,0), limits=c(1.5, 5.1)) +
  ggtitle("Power law fits for (r,p) graphs") +
  my_theme(11) -> p
ggsave('../results/fig_2a.pdf', p, width=4.5, height=2.5)

DF %>% filter(type == 'g') %>%
ggplot(aes(x=r_off, y=mean_r, color=p)) + geom_line() +
  geom_segment(aes(x=r_off, xend=r_off, y=ll_r, yend=ul_r)) +  # too wide, could offset x-axis slightly
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $r$"), expand=c(0,0), limits=c(-0.505, 0.4)) +
  ggtitle("Jackson r fits for (r,p) graphs") +
  my_theme(11) -> p
ggsave('../results/fig_2b.pdf', p, width=4.5, height=2.5)

ggplot(DF, aes(x=r, y=1/(mean_a-1), color=p)) + geom_line() +
  #geom_segment(aes(x=r_off, xend=r_off, y=1/(ll_a-1), yend=1/(ul_a-1))) +
  geom_hline(yintercept=1, color='grey', linetype='dashed') + geom_line() +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of p"), expand=c(0,0), limits=c(0, 1.05)) +#, breaks=c(0.6, 0.8, 1.0, 1.2)) +
  #ggtitle("Degree distribution fits for (r,p) graphs") +
  facet_grid(~type) +
  my_theme(11) -> p
ggsave('../results/fig_2.pdf', p, width=4.5, height=2.5)



##
## Figure 3 - Attachment function comparing Newman,Pham,degree-model
##

# # Python code to generate a PA graph with m=1
# import util, logit, synth_generate, synth_process, csv
# (G, el) = synth_generate.make_rp_graph('test', n_max=25000, r=1, p=.99, m=1, grow=True)
# fn = '%s/synth_graphs/test_pa.csv' % util.data_path
# synth_generate.write_edge_list(el, fn)
# synth_process.process_all_edges('test_pa.csv', n_alt=20, vvv=0)
# m = logit.DegreeModel('test_pa.csv', vvv=1, max_deg=100)
# m.fit()
# with open("../results/deg_fit.csv", 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['deg','coef','se'])
#     for i in range(len(m.u)):
#         x = writer.writerow([i, m.u[i], m.se[i]])
d <- read_csv("../data/choices_grouped/test_pa.csv", col_types='iiii')
DF <- d %>%
  filter(c==1) %>%
  inner_join(d %>% group_by(choice_id) %>% summarize(tot=sum(n)), by='choice_id') %>%
  mutate(p=n/tot) %>%
  group_by(deg) %>%
  summarize(stat=sum(1/p)) %>%
  left_join(d %>% group_by(deg) %>% summarize(w=1/n()), by='deg')

DF <- rbind(
    DF %>% select(deg, stat) %>% mutate(type='Newman [36]'),
    DF %>% mutate(stat=stat*w) %>% select(deg,stat) %>% mutate(type='Newman\nCorrected [39]'),
    read_csv("../results/deg_fit.csv", col_types='idd') %>%
      select(deg, stat=coef) %>% mutate(stat=exp(stat), type='Conditional\nLogit'))

DF %>%
  inner_join(DF %>% filter(deg==1) %>% select(type, ref=stat), by=c('type')) %>%
  filter(deg > 0, deg < 101) %>%
  mutate(
    stat=stat/ref,
    type=factor(type, levels=c('Conditional\nLogit','Newman [36]','Newman\nCorrected [39]'))
  ) %>%
  ggplot(aes(deg, stat, color=type)) + geom_point(shape=20, alpha=0.7) +
    scale_x_log10("log Degree", limits=c(1, 100), labels=trans_format('log10', math_format(10^.x)), breaks=c(10^0, 10^1, 10^2)) +
    scale_y_log10("Relative likelihood", limits=c(1, 110), labels=trans_format('log10', math_format(10^.x))) +
    scale_color_brewer(palette='Set1') + 
    my_theme() + theme(legend.title=element_blank(), legend.position=c(0.14, 0.78)) -> p
ggsave('../results/fig_3.pdf', p, width=4, height=2.5)
