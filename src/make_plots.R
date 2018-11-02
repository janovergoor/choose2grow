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
library(gridExtra)

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
  scale_fill_gradientn(colours = heat.colors(6)) +
  geom_line(data=em, color='black') +
  scale_x_continuous(TeX("$\\alpha$"), limits=c(0, 2), expand=c(0,0)) +
  scale_y_continuous(TeX("$\\pi_1$"), limits=c(0, 1), expand=c(0,0)) +
  geom_point(data=data.frame(x=1, y=0.5), shape='x', size=4) +
  geom_point(data=em %>% head(n=1), shape=1 , size=3) +
  geom_point(data=em %>% tail(n=1), shape=16, size=3) +
  my_theme() + theme(legend.position='none',
                     axis.line = element_blank(), panel.border = element_blank(),
                     panel.background = element_blank()) -> p

ggsave('../results/fig_1.pdf', p, width=4.5, height=2.5)





##
## Figure 2 - Attachment function comparing Newman,Pham,degree-model
##

# read edge stats
DF <- read_csv("../data/choices_grouped/test_pa.csv", col_types='iiii')
# compute newman attachment function
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
# normalize by coefficient for degree 1
DFnp$stat = DFnp$stat / DFnp[DFnp$deg==1, ]$stat
# least squares fit of normalized coefficients
fit2 <- lm(stat ~ 0 + deg, weights=w, data=DFnp %>% filter(coef!=1))

# join results together
DF <- rbind(
  # Newman
  DF %>% select(deg, stat) %>% mutate(id='newman', label='Newman', ll=0, ul=0),
  # Newman corrected
  DF %>% mutate(stat=stat*w) %>% select(deg,stat) %>% mutate(id='newman2', label='Newman\nCorrected', ll=0, ul=0),
  # non-parametric coefficients
  DFnp %>% select(deg, stat) %>% mutate(id='npl', label='Non-parametric\nlogit', ll=0, ul=0),
  # create predicted values of least squares fit
  predict(fit2, data.frame(deg=1:100), interval="confidence") %>% as.data.frame() %>% mutate(deg=1:100, id='ls', label='Least-squares') %>%
    select(deg, stat=fit, id, label, ll=lwr, ul=upr),
  # compute log-degree logit fit
  data.frame(deg=1:100) %>% mutate(stat=deg^(fit1$coef), id='ldl', label='Log-degree\nlogit', ll=deg^(fit1$coef - 1.96*fit1$se), ul=deg^(fit1$coef + 1.96*fit1$se))
  )

# normalize by degree 1
DF <- DF %>% filter(id %in% c('ldl', 'ls', 'npl')) %>% mutate(ref=1) %>%
  rbind(
    DF %>% filter(id %in% c('newman', 'newman2')) %>%
      inner_join(DF %>% filter(deg==1) %>% select(id, ref=stat), by=c('id'))
  ) %>%
  filter(deg > 0, deg < 101) %>%
  filter(id != 'newman2') %>%
  mutate(
    stat=stat/ref,
    label=factor(label, levels=c('Newman','Non-parametric\nlogit','Least-squares','Log-degree\nlogit'))
  )

ggplot(DF, aes(deg, stat, color=label)) +
  geom_point(shape=20, alpha=0.0) +
  geom_line(data=DF %>% filter(id=='ls'), size=0.5) +
  geom_line(data=DF %>% filter(id=='ldl'), size=0.5) +
  geom_point(data=DF %>% filter(id=='newman'), shape=20, alpha=0.7) +
  geom_point(data=DF %>% filter(id=='npl'), shape=20, alpha=0.7) +
  scale_x_log10("log Degree", labels=trans_format('log10', math_format(10^.x)), breaks=c(10^0, 10^1, 10^2), expand=c(0,0)) +
  scale_y_log10("Relative likelihood", labels=trans_format('log10', math_format(10^.x)), expand=c(0,0)) +
  coord_cartesian(xlim=c(1, 100), ylim=c(1, 100)) +
  scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#CC79A7"),
                     guide = guide_legend(override.aes=list(
                       linetype=c("blank", "blank", "solid", "solid"), shape=c(16, 16, NA, NA)))) +
    my_theme() + theme(legend.title=element_blank(),
                     axis.line = element_line(colour="black"), panel.border=element_blank(), panel.background=element_blank()) -> p

ggsave('../results/fig_2.pdf', p, width=4.5, height=2.5)


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
  }, mc.cores=5) %>%
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

write_csv(DF, "../results/fig3_data.csv")

read_csv("../results/fig3_data.csv", col_types='ccccdddddddccd') %>%
  filter(type=='g', type2=='u') %>%
  ggplot(aes(x=r_off, y=mean_a, color=p)) + geom_line() + geom_point(show.legend=F) +
    #geom_segment(aes(x=r_off, xend=r_off, y=ll_a, yend=ul_a)) +
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00")) + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous(TeX("Estimate of $\\gamma$"), expand=c(0,0), limits=c(2, 5.1)) +
    geom_hline(yintercept=3, color='grey', linetype='dashed') +
    my_theme(11) + theme(legend.title=element_text(hjust=0.5),
      axis.line=element_line(colour="black"), panel.border=element_blank(), panel.background=element_blank()) -> p
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
  geom_hline(yintercept=1, color='grey', linetype='dashed') + geom_line() +
  scale_color_brewer(palette='Set1') + 
  scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous(TeX("Estimate of p"), expand=c(0,0), limits=c(0, 1.05)) +
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
  mutate(Model=ifelse(model=='p', 'Copy\nmodel', 'Local\nsearch'))

DF_max <- DF %>% group_by(data, Model) %>% filter(ll==max(ll))

pchisq(-2*log(DF_max$ll[4]/DF_max$ll[3]), df=1, lower.tail=F)
pchisq(-2*log(DF_max$ll[1]/DF_max$ll[2]), df=1, lower.tail=F)
pchisq(2 * abs(DF_max$ll[4]-DF_max$ll[3]), df=1, lower.tail=F)
pchisq(2 * abs(DF_max$ll[2]-DF_max$ll[1]), df=1, lower.tail=F)



p1 <- DF %>% filter(data=='r=0.50, p=1.00') %>%
  ggplot(aes(p, ll, color=Model)) + geom_line() +
  geom_point(data=DF_max %>% filter(data=='r=0.50, p=1.00'), show.legend=F) +
  scale_x_continuous("Class probability", labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous("Log-likelihood", limits=c(-40000,-25000), labels=function(x) sprintf("%.0fk", x/1000)) +
  scale_color_manual(values = c("#E69F00", "#56B4E9")) +
  ggtitle("r=0.50, p=1.00") +
  my_theme() + theme(legend.position='none',                        
                     axis.line = element_line(colour = "black"), panel.border=element_blank(), panel.background=element_blank())

p2 <- DF %>% filter(data=='r=1.00, p=0.50') %>%
  ggplot(aes(p, ll, color=Model)) + geom_line() +
  geom_point(data=DF_max %>% filter(data=='r=1.00, p=0.50'), show.legend=F) +
  scale_x_continuous("Class probability", labels=c('0','0.25','0.50','0.75','1')) +
  scale_y_continuous("Log-likelihood", limits=c(-43000, -35000), labels=function(x) sprintf("%.0fk", x/1000)) +
  scale_color_manual(values = c("#E69F00", "#56B4E9")) +
  ggtitle("r=1.00, p=0.50") +
  my_theme() + theme(legend.title=element_blank(), legend.position=c(0.85, 0.19), axis.title.y=element_blank(),
                     axis.line = element_line(colour = "black"), panel.border=element_blank(), panel.background=element_blank())

pdf('../results/fig_4.pdf', width=6, height=3)
lay <- rbind(c(rep(1, 10), rep(2, 9)),
             c(rep(1, 10), rep(2, 9)))
grid.arrange(p1, p2, layout_matrix = lay)
dev.off()




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
  geom_line(aes(linetype=Type2)) +
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


##
## Figure 6 - Non-parametric per model
##
DF <- rbind(
    read_csv("~/projects/choosing_to_grow/choose2grow/results/fig6_flickr.csv", col_types='ddcc') %>%
      mutate(Model=paste0('3.', model), data='Flickr'),
    read_csv("~/projects/choosing_to_grow/choose2grow/results/fig6_citations.csv", col_types='ddcc') %>%
      mutate(Model=paste0('4.', model), data='Citations')
  )  %>%
  filter(!(type=='line' & deg==0)) %>%
  mutate(deg=ifelse(deg==0, 0.5, deg))

DF <- DF %>%
  inner_join(DF %>% filter(type=='line', deg==1) %>% select(Model, data, est1=est)) %>% mutate(est=est/est1) %>%
  mutate(data=factor(data, levels=c('Flickr','Citations')))

p1 <- DF %>% filter(type=='point', data=='Flickr') %>%
  ggplot(aes(deg, est, color=Model)) + 
  geom_line(alpha=0) +
  geom_line(data=data.frame(deg=1:100, est=1:100), color='black', linetype='dashed') +
  geom_point(shape=16, size=.8, show.legend=F) +
  geom_point(data=DF %>% filter(type=='point', deg==0.5, data=='Flickr'), shape=16, size=1.2, show.legend=F) +
  geom_line(data=DF %>% filter(type=='line', data=='Flickr'), alpha=0.6) +
  scale_x_log10('Degree', breaks=c(0.5, 1, 10, 100), labels=c(TeX("$\\,0^{\\;}$"), TeX("$10^0$"), TeX("$10^1$"), TeX("$10^2$"))) +
  scale_y_log10('Relative Probability', labels=trans_format('log10', math_format(10^.x)), limits=c(0.2, 250)) +
  guides(colour=guide_legend(override.aes=list(alpha = 1))) +
  scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73")) +
  ggtitle("Flickr") +
  my_theme(10) + theme(legend.position=c(0.89, 0.18), legend.title=element_blank(),
                       axis.line = element_line(colour = "black"), panel.border = element_blank(), panel.background = element_blank())

p2 <- DF %>% filter(type=='point', data=='Citations') %>%
  ggplot(aes(deg, est, color=Model)) + 
  geom_line(alpha=0) +
  geom_line(data=data.frame(deg=1:100, est=1:100), color='black', linetype='dashed') +
  geom_point(shape=16, size=.8, show.legend=F) +
  geom_point(data=DF %>% filter(type=='point', deg==0.5, data=='Citations'), shape=16, size=1.2, show.legend=F) +
  geom_line(data=DF %>% filter(type=='line', data=='Citations'), alpha=0.6) +
  scale_x_log10('Degree', breaks=c(0.5, 1, 10, 100), labels=c(TeX("$\\,0^{\\;}$"), TeX("$10^0$"), TeX("$10^1$"), TeX("$10^2$"))) +
  scale_y_log10('Relative Probability', labels=trans_format('log10', math_format(10^.x)), limits=c(0.2, 250)) +
  guides(colour=guide_legend(override.aes=list(alpha = 1))) +
  scale_color_manual(values = c("#E69F00", "#56B4E9")) +
  ggtitle("Citations") +
  my_theme(10) + theme(legend.position=c(0.89, 0.13), legend.title=element_blank(),
                       axis.title.y=element_blank(), axis.text.y=element_blank(),
                       axis.line = element_line(colour = "black"), panel.border = element_blank(), panel.background = element_blank()) -> p2

pdf('../results/fig_6.pdf', width=6, height=3)
lay <- rbind(c(rep(1, 10), rep(2, 9)),
             c(rep(1, 10), rep(2, 9)))
grid.arrange(p1, p2, layout_matrix = lay)
dev.off()
