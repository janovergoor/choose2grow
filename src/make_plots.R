# Run with Rscript make_plots.R

library(dplyr)
library(ggplot2)
library(latex2exp)
library(stringr)
library(parallel)
library(tidyr)
library(readr)
library(scales)


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

list.files("../data/graphs", pattern='g.*') %>%
  mclapply(function(fn){
    el <- read_csv(paste0("../data/graphs/", fn), col_types='iii')
    el <- rbind(data.frame(i=el$from, j=el$to),
                data.frame(i=el$to, j=el$from)) %>%
      group_by(i) %>% summarize(deg=n())
    fit <- plfit(el$deg, "range", seq(1.001,5,0.01))
    vals = str_split(fn, '-')[[1]]
    data.frame(type=vals[1], r=vals[2], p=vals[3], id=substr(vals[4], 1, 2), alpha=fit$alpha, xmin=fit$xmin)
  }, mc.cores=10) %>%
  bind_rows() %>%
  group_by(type, r, p) %>%
  summarize(mean=mean(alpha), ll=min(alpha), ul=max(alpha)) %>% ungroup() %>%
  mutate(r=as.numeric(r)) %>%
  ggplot(aes(x=r, y=mean, color=p)) + geom_line() +
    #geom_segment(aes(x=r, xend=r, y=ll, yend=ul)) +  # too wide, could offset x-axis slightly
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of '$\\alpha$'"), expand=c(0,0), limits=c(2.5, 5.1)) + #, breaks=seq(0, 1, 0.25)) +
    ggtitle("Power law fits for (r,p) graphs") +
    my_theme(11) -> p3

ggsave('../results/fig_3.pdf', p3, width=4.5, height=2.5)



##
## Figure 4 - Attachment function comparing Newman,Pham,degree-model
##

# import logit
# m = logit.DegreeModel('g-1.00-1.00-00.csv', vvv=1, max_deg=100)
# m.fit()
man = c(0, 0, 0, 0, -1.61597202161, -1.35146438385, -1.12757800016, -1.08591613752, -0.919866712021, -0.887493592372, -0.783775116436, -0.595905754175, -0.55642965309, -0.400138996292, -0.331475596958, -0.296699035766, -0.234320744108, -0.124624152867, -0.110778169014, -0.0949602984552, 0.013620747728, 0.101275808782, 0.0572693092161, 0.0936274545126, 0.56574206998, 0.0344010884704, 0.251416551657, 0.375346768283, 0.344526188077, 0.333147333171, 0.87347730392, 0.740368725218, 0.168113118535, 0.939589767518, 0.536955797355, 0.820818624756, 0.598112673987, 0.345560164076, 0.698238319183, 0.491818620288, 0.885320821118, 1.0053390944, 1.14708025477, 0.893489550398, 0.857973423636, 0.811982738348, 0.848205151958, 0.635641015481, 0.858738752628, 0.932459483865, 0.527799768823, 1.01427849536, 1.08470768056, 1.38604760356, 0.948996516731, 1.23591747035, 1.25257984284, 0.952277252416, 1.08516070237, 2.28020352105, 1.03136370952, 1.30811668004, 1.63912613349, 1.50030321144, 1.34738920555, 0.842481822644, 1.03030742903, 1.23210793395, 0.798712113096, 1.96943920468, 0.342585331651, 1.08254496118, 0.997659541365, 2.08148956987, 0.60026913808, 1.1886975219, 1.39918805379, 1.39076011003, 1.42839318122, 0.303665495085, 1.53600005811, 0.344483698076, 0.900486585841, 1.87863117696, 1.28570383682, 1.17365370842, 1.38515806559, 1.5215047931, 3.04073485159, 1.87351425053, 1.00794757526, 1.62449883055, 1.82418562824, 3.26829975092, 24.93706832, 2.13068008983, 0.932791571555, 1.92011970084, 0.746668249404, 2.35192260815, 1.43177814701)
man = man - min(man) + 1

p = 1
df = read_csv(sprintf("../data/choices_grouped/g-1.00-%.2f-00.csv", p), col_types='iiii') %>%
  filter(c==1) %>%
  inner_join(d %>% group_by(choice_id) %>% summarize(tot=sum(n)), by='choice_id') %>%
  mutate(p=n/tot) %>%
  group_by(deg) %>%
  summarize(stat=sum(1/p)) %>%
  left_join(d %>% group_by(deg) %>% summarize(w=1/n()), by='deg')

df = rbind(
    df %>% select(deg, stat) %>% mutate(type='Newman'),
    df %>% mutate(stat=stat*w) %>% select(deg,stat) %>% mutate(type='Pham'),
    data.frame(deg=0:100, stat=man, type='Logit'))

df %>%
  inner_join(df %>% filter(deg==4) %>% select(type, ref=stat), by=c('type')) %>%
  filter(deg > 3, deg < 101) %>%
  mutate(stat=stat/ref) %>%
  ggplot(aes(deg, stat, color=type)) + geom_point(shape=20, alpha=0.7) +
  scale_x_log10("log Degree", limits=c(4, 100), labels=trans_format('log10', math_format(10^.x))) +
  scale_y_log10("Relative likelihood", limits=c(1, 110), labels=trans_format('log10', math_format(10^.x))) +
  ggtitle("Attachment Function") +
  scale_color_brewer(palette='Set1') + 
  my_theme() + theme(legend.title=element_blank()) -> p4

ggsave('../results/fig_4.pdf', p4, width=4.5, height=2.5)

