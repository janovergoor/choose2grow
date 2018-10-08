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
        Type=ifelse(type=='g', 'Growth', 'Densify'),
        r=as.numeric(r),
        P=paste0('p=', p)
      )

DF %>%
  filter(Model != 'p-single', type=='g') %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, mean, color=p)) +
    geom_line() + #geom_point() +
    geom_segment(aes(x=r, xend=r, y=ll, yend=ul)) +
    facet_grid( ~ Model) +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of $p$"), expand=c(0,0), limits=c(0, 1.02), breaks=seq(0, 1, 0.25)) +
    #scale_color_manual(values=c("#fdd49e","#fc8d59","#ef6548","#d7301f","#b30000")) +  # orange
    #scale_color_manual(values=c("#d0d1e6","#a6bddb","#3690c0","#0570b0","#045a8d")) +  # blue
    my_theme(11) -> p1

ggsave('../results/fig_1.pdf', p1, width=6, height=3)



##
## Figure 2 - LL by PA and PA-FoF models
## currently not in the paper
##

DF %>%
  filter(Model != 'p-single') %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, LL, color=Model)) +
    geom_line() +
    facet_grid(Type ~ P, scales='free_y') +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous("Log-likelihood") +
    my_theme(11) -> p2

ggsave('../results/fig_2.pdf', p2, width=9, height=3.5)



##
## Figure 3 - Power law fits on degree of (r,p) graphs
##

source("http://tuvalu.santafe.edu/~aaronc/powerlaws/plfit.r")

DF <- list.files("../data/graphs", pattern='g.*') %>%
  mclapply(function(fn){
    el <- read_csv(paste0("../data/graphs/", fn), col_types='iii')
    el <- rbind(data.frame(i=el$from, j=el$to),
                data.frame(i=el$to, j=el$from)) %>%
      group_by(i) %>% summarize(deg=n())
    # compute powerlaw fit
    fit <- plfit(el$deg, "range", seq(1.001,5,0.01))
    # compute jacksons r
    cdf <- el %>% group_by(deg) %>% summarize(n=n()) %>% ungroup() %>%
      arrange(deg) %>% complete(deg=seq(max(deg)), fill=list(n=0)) %>%
      mutate(F_d=cumsum(n)/sum(n))
    rstar = r_jackson(cdf$F_d, mean(el$deg))
    r_est = rstar/(1+rstar)
    # gather results
    vals <- str_split(fn, '-')[[1]]
    data.frame(type=vals[1], r=vals[2], p=vals[3], id=substr(vals[4], 1, 2),
               alpha=fit$alpha, xmin=fit$xmin, r_est=r_est)
  }, mc.cores=10) %>%
  bind_rows() %>%
  group_by(type, r, p) %>%
  summarize(mean_a=mean(alpha), ll_a=quantile(alpha, 0.25), ul_a=quantile(alpha, 0.75),
            mean_r=mean(r_est), ll_r=quantile(r_est, 0.25), ul_r=quantile(r_est, 0.75)
            ) %>% ungroup() %>%
  mutate(r=as.numeric(r)) %>%
  # offset r slightly
  mutate(r_off = r + (as.numeric(p)-0.5)/50)

ggplot(DF, aes(x=r_off, y=mean_a, color=p)) + geom_line() +
  geom_segment(aes(x=r_off, xend=r_off, y=ll_a, yend=ul_a)) +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $\\gamma$"), expand=c(0,0), limits=c(2.5, 5.1)) +
  ggtitle("Power law fits for (r,p) graphs") +
  my_theme(11) -> p3a
ggsave('../results/fig_3a.pdf', p3a, width=4.5, height=2.5)

ggplot(DF, aes(x=r_off, y=mean_r, color=p)) + geom_line() +
  geom_segment(aes(x=r_off, xend=r_off, y=ll_r, yend=ul_r)) +  # too wide, could offset x-axis slightly
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $r$"), expand=c(0,0), limits=c(0.05, 0.35)) +
  ggtitle("Jackson r fits for (r,p) graphs") +
  my_theme(11) -> p3b
ggsave('../results/fig_3b.pdf', p3b, width=4.5, height=2.5)

ggplot(DF, aes(x=r, y=2/(mean_a-1), color=p)) + geom_line() +
  #geom_segment(aes(x=r_off, xend=r_off, y=2/(ll_a-1), yend=2/(ul_a-1))) +
  geom_hline(yintercept=1, color='grey', linetype='dashed') + geom_line() +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of p"), expand=c(0,0), limits=c(0.45, 1.3), breaks=c(0.6, 0.8, 1.0, 1.2)) +
  #ggtitle("Degree distribution fits for (r,p) graphs") +
  my_theme(11) -> p3
ggsave('../results/fig_3.pdf', p3, width=4.5, height=2.5)



##
## Figure 4 - Attachment function comparing Newman,Pham,degree-model
##

# import logit
# m = logit.DegreeModel('g-1.00-1.00-00.csv', vvv=1, max_deg=100)
# m.fit()
# for i in range(len(m.u)):
#     print("%d,%f,%f" % (i, m.u[i], m.se[i]))
logit <- read_csv("../results/deg_fit.csv", col_types='idd')
# adjust such that k=4 is the reference point
logit$coef <- logit$coef + abs(logit$coef[5]) + 1

p <- 1
d <- read_csv(sprintf("../data/choices_grouped/g-1.00-%.2f-00.csv", p), col_types='iiii')
DF <- d %>%
  filter(c==1) %>%
  inner_join(d %>% group_by(choice_id) %>% summarize(tot=sum(n)), by='choice_id') %>%
  mutate(p=n/tot) %>%
  group_by(deg) %>%
  summarize(stat=sum(1/p)) %>%
  left_join(d %>% group_by(deg) %>% summarize(w=1/n()), by='deg')

DF <- rbind(
    DF %>% select(deg, stat) %>% mutate(type='Newman'),
    DF %>% mutate(stat=stat*w) %>% select(deg,stat) %>% mutate(type='Pham et al.'),
    logit %>% select(deg, stat=coef) %>% mutate(type='Logit'))

DF %>%
  inner_join(DF %>% filter(deg==4) %>% select(type, ref=stat), by=c('type')) %>%
  filter(deg > 3, deg < 101) %>%
  mutate(stat=stat/ref) %>%
  ggplot(aes(deg, stat, color=type)) + geom_point(shape=20, alpha=0.7) +
    scale_x_log10("log Degree", limits=c(4, 100), labels=trans_format('log10', math_format(10^.x)), breaks=c(10^1, 10^1.5, 10^2)) +
    scale_y_log10("Relative likelihood", limits=c(1, 110), labels=trans_format('log10', math_format(10^.x))) +
    scale_color_brewer(palette='Set1') + 
    my_theme() + theme(legend.title=element_blank()) -> p4

ggsave('../results/fig_4.pdf', p4, width=4.5, height=2.5)

