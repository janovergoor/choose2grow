# Run with Rscript make_plots.R

library(dplyr)
library(ggplot2)
library(latex2exp)
library(stringr)
library(parallel)
library(tidyr)
library(readr)
library(scales)
library(dplyr)

setwd("~/projects/choosing_to_grow/choose2grow/src")
source('../reports/helper.R')

##
## Figure 1 - Likelihood surface
##
DF <- read_csv("../results/fig1_data.csv", col_types='ddd') %>%
  # weird but monotomic transformation of ll to get the colors right
  mutate(ll=exp((ll+55000)/1000)) %>%
  select(x=alpha, y=p, z=ll)
em = read_csv("../results/fig1_data_em.csv", col_types='iddddd') %>%
  mutate(x=u2, y=p1)

ggplot(DF, aes(x, y)) +
  geom_raster(aes(fill=z), interpolate=T) +
  geom_contour(aes(z=z), colour="black", bins=15, alpha=0.25) +
  #stat_contour(geom="polygon", aes(z=z, fill=..level.. ), bins=15) +
  #scale_fill_gradient2(low="#d73027", mid='#fee090', high="#4575b4", midpoint=2000) + # red:blue
  scale_fill_gradientn(colours = heat.colors(6)) +
  geom_line(data=em, color='black') +
  scale_x_continuous(TeX("$\\alpha$"), limits=c(0, 2), expand=c(0,0)) +
  scale_y_continuous(TeX("$\\pi_1$"), limits=c(0, 1), expand=c(0,0)) +
  geom_point(data=data.frame(x=1, y=0.5), shape='x', size=4) +
  geom_point(data=em %>% head(n=1), shape=1 , size=2) +
  geom_point(data=em %>% tail(n=1), shape=16, size=2) +
  my_theme() + theme(legend.position='none') -> p

ggsave('../results/fig_1.pdf', p, width=4.5, height=3)



##
## Figure 2 - Attachment function comparing Newman,Pham,degree-model
##

DF <- read_csv("../data/choices_grouped/test_pa.csv", col_types='iiii')
DF <- DF %>%
  filter(c==1) %>%
  inner_join(DF %>% group_by(choice_id) %>% summarize(tot=sum(n)), by='choice_id') %>%
  mutate(p=n/tot) %>%
  group_by(deg) %>%
  summarize(stat=sum(1/p)) %>%
  left_join(DF %>% group_by(deg) %>% summarize(w=1/n()), by='deg')

# read log-degree logit fit
fit1 <- read_csv("../results/fig2_data.csv", col_types='cdd') %>% filter(deg == 'alpha')
# read non-parametric degree log fit
DFnp <- read_csv("../results/fig2_data.csv", col_types='cdd') %>% filter(deg != 'alpha') %>%
  filter(coef<20) %>%
  mutate(deg=as.numeric(deg), stat=exp(coef), w=1/se)
DFnp$stat = DFnp$stat / DFnp[DFnp$deg==1, ]$stat
fit2 <- lm(stat ~ 0 + deg, weights=w, data=DFnp)

DF <- rbind(
  # compute Newman
  DF %>% select(deg, stat) %>% mutate(id='newman', label='Newman [41]', ll=0, ul=0),
  # compute Newman corrected
  DF %>% mutate(stat=stat*w) %>% select(deg,stat) %>% mutate(id='newman2', label='Newman\nCorrected [44]', ll=0, ul=0),
  # read non-parametric degree logit fits
  DFnp %>% select(deg, stat) %>% mutate(id='npl', label='Individual degree logit', ll=0, ul=0),
  predict(fit2, data.frame(deg=1:100), interval="confidence") %>% as.data.frame() %>% mutate(deg=1:100, id='ls', label='Least-squares [44]') %>%
    select(deg, stat=fit, id, label, ll=lwr, ul=upr),
  # compute log-degree logit fit
  data.frame(deg=1:100) %>% mutate(stat=deg^fit1$coef, id='ldl', label='Log-degree logit', ll=deg^(fit1$coef - 1.96*fit1$se), ul=deg^(fit1$coef + 1.96*fit1$se))
  )

DF <- DF %>% filter(id %in% c('ldl', 'ls', 'npl')) %>% mutate(ref=1) %>%
  rbind(
    DF %>% filter(id %in% c('newman', 'newman2')) %>%
      inner_join(DF %>% filter(deg==1) %>% select(id, ref=stat), by=c('id'))
  ) %>%
  filter(deg > 0, deg < 101) %>%
  filter(id != 'newman2') %>%
  mutate(
    stat=stat/ref,
    label=factor(label, levels=c('Newman [41]','Individual degree logit','Least-squares [44]','Log-degree logit'))
  )

ggplot(DF, aes(deg, stat, color=label)) +
  geom_point(shape=20, alpha=0.0) +
  #geom_abline(slope=1, intercept=0, color='grey') +
  geom_line(data=DF %>% filter(id=='ls'), show.legend=F, size=0.5) +
  geom_line(data=DF %>% filter(id=='ldl'), show.legend=F, size=0.5) +
  geom_point(data=DF %>% filter(id=='newman'), shape=20, alpha=0.7) +
  geom_point(data=DF %>% filter(id=='npl'), shape=20, alpha=0.7) +
  scale_x_log10("log Degree", labels=trans_format('log10', math_format(10^.x)), breaks=c(10^0, 10^1, 10^2), expand=c(0,0)) +
  scale_y_log10("Relative likelihood", labels=trans_format('log10', math_format(10^.x)), expand=c(0,0)) +
  coord_cartesian(xlim=c(1, 100), ylim=c(1, 120)) +
  scale_color_brewer(palette='Set1') + 
  my_theme() + theme(legend.position=c(0.20, 0.79), legend.title=element_blank()) -> p

ggsave('../results/fig_2.pdf', p, width=4, height=3)


##
## Figure 3 - Power law fits on degree of (r,p) graphs
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
    data.frame(type=vals[1], r=vals[2], p=vals[3], type2=vals[4], id=substr(vals[5], 1, 2),
               alpha=fit$alpha, xmin=fit$xmin, r_est=r_est)
  }, mc.cores=10) %>%
  bind_rows() %>%
  group_by(type, r, p, type2) %>%
  summarize(mean_a=mean(alpha), ll_a=quantile(alpha, 0.25), ul_a=quantile(alpha, 0.75),
            mean_r=mean(r_est), ll_r=quantile(r_est, 0.25), ul_r=quantile(r_est, 0.75)
  ) %>% ungroup() %>%
  mutate(
    r=as.numeric(r),
    # offset r slightly
    r_off = r + (as.numeric(p)-0.5)/50,
    Type=ifelse(type=='g', 'External / Growth', 'Internal / Densify'),
    Type2=ifelse(type2=='d', 'Directed', 'Undirected'),
    # compute p_hat
    p_hat=ifelse(type2=='d', (mean_a-2)/(mean_a-1), (mean_a-3)/(mean_a-1))
  )


DF %>%
  filter(type=='g', type2=='u') %>%
  ggplot(aes(x=r_off, y=mean_a, color=p)) + geom_line() +
    geom_segment(aes(x=r_off, xend=r_off, y=ll_a, yend=ul_a)) +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of $\\gamma$"), expand=c(0,0), limits=c(2, 5.1)) +
    geom_hline(yintercept=3, color='lightgrey', linetype='dashed') +
    my_theme(11) -> p
ggsave('../results/fig_3.pdf', p, width=4.5, height=2.5)


ggplot(DF, aes(x=r_off, y=mean_a, color=p)) + geom_line() +
  geom_segment(aes(x=r_off, xend=r_off, y=ll_a, yend=ul_a)) +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $\\gamma$"), expand=c(0,0), limits=c(1.5, 5.1)) +
  ggtitle("Power law fits for (r,p) graphs") + facet_grid(Type2 ~ Type) +
  geom_hline(data=data.frame(y=c(2,3), Type2=c('Directed','Undirected')), aes(yintercept=y), color='lightgrey', linetype='dashed') +
  my_theme(11) -> p
ggsave('../results/fig_3_si1.pdf', p, width=4.5, height=3.5)

ggplot(DF, aes(x=r, y=p_hat, color=p)) + geom_line() +
  #geom_segment(aes(x=r_off, xend=r_off, y=1/(ll_a-1), yend=1/(ul_a-1))) +
  geom_hline(yintercept=1, color='grey', linetype='dashed') + geom_line() +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of p"), expand=c(0,0), limits=c(0, 1.05)) +#, breaks=c(0.6, 0.8, 1.0, 1.2)) +
  ggtitle("p fits for (r,p) graphs") +
  facet_grid(Type2 ~ Type) +
  my_theme(11) -> p
ggsave('../results/fig_3_si2.pdf', p, width=4.5, height=3.5)

ggplot(DF, aes(x=r_off, y=mean_r, color=p)) + geom_line() +
  geom_segment(aes(x=r_off, xend=r_off, y=ll_r, yend=ul_r)) +  # too wide, could offset x-axis slightly
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $r$"), expand=c(0,0), limits=c(-0.5, 0.5)) +
  ggtitle("Jackson r fits for (r,p) graphs") + facet_grid(Type2 ~ Type) +
  my_theme(11) -> p
ggsave('../results/fig_3_si3.pdf', p, width=4.5, height=3.5)


##
## Figure 4 - LL of wrong models
##

DF <- read_csv("../results/fig4_data.csv", col_types='ccdd') %>%
  mutate(Model=ifelse(model=='p', 'Copy\nmodel', 'Jackson\nRogers'))

DF %>%
  ggplot(aes(p, ll, color=Model)) + geom_line() +
  geom_point(data=DF %>% group_by(data, Model) %>% filter(ll==max(ll)), show.legend=F) +
  facet_wrap(~data, scales='free_y') +
  scale_x_continuous("Class probability", expand=c(0,0)) +
  scale_y_continuous("Log-likelihood", limits=c(-45000,NA),
                     labels=function(x) sprintf("%.0fk", x/1000)) +
  my_theme() + theme(legend.position=c(0.90, 0.19), legend.title=element_blank()) -> p

ggsave('../results/fig_4.pdf', p, width=4.5, height=2.5)


##
## Figure 5 - estimates of p by PA and PA-FoF models
##
DF <- read_csv("../results/fig5_data.csv", col_types='ccdd') %>%
  separate(fn, into=c("type","r","p","type2","id"), sep='-') %>%
  filter(p != '0.00', p != '0.99') %>%
  group_by(type, r, p, type2, model) %>%
  summarize(LL=-1*mean(ll), mean=mean(estimate),
            ll=min(estimate), ul=max(estimate),
            n=n()) %>%
  ungroup() %>%
  mutate(
    Model=model,
    Type=ifelse(type=='g', 'External / Growth', 'Internal / Densify'),
    Type2=ifelse(type2=='d', 'Directed', 'Undirected'),
    r=as.numeric(r),
    P=paste0('p=', p)
  )

DF %>%
  filter(type=='g', type2=='u') %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, mean, color=p)) +
    geom_line() +
    geom_segment(aes(x=r, xend=r, y=ll, yend=ul)) +
    facet_grid(~Model) +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of $p$"), expand=c(0,0), limits=c(0, 1.02), breaks=seq(0, 1, 0.25)) +
    my_theme(11) -> p
ggsave('../results/fig_5.pdf', p, width=6, height=3)

## Figure 5a - all model estimates
DF %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, mean, color=p)) +
  geom_line(aes(linetype=Type2)) + #geom_point() +
  geom_segment(aes(x=r, xend=r, y=ll, yend=ul)) +
  facet_grid(Type ~ Model) +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of $p$"), expand=c(0,0), limits=c(0, 1.02), breaks=seq(0, 1, 0.25)) +
  my_theme(11) -> p
ggsave('../results/fig_5_si1.pdf', p, width=6, height=4)

## Figure 5b - LL by PA and PA-FoF models
DF %>%
  filter(type2=='u') %>%
  ggplot(aes(r, LL, color=Model)) +
  geom_line() +
  facet_grid(Type ~ P, scales='free_y') +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous("Log-likelihood") +
  my_theme(11) -> p
ggsave('../results/fig_5_si2.pdf', p, width=9, height=3.5)
