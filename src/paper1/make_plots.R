# Run with `Rscript paper1/make_plots.R`

source('helper.R')
data_folder = '../data/paper1_plot_data/'
figures_folder = '../figures/paper1/'

##
## Figure 1 - Likelihood surface
##

DF <- sprintf("%s/%s", data_folder, "fig_1_data.csv") %>%
  read_csv(col_types='ddd') %>%
  # weird but monotomic transformation of ll to get the colors right
  mutate(ll=exp((ll+55000)/1000)) %>%
  select(x=alpha, y=p, z=ll)

em <- sprintf("%s/%s", data_folder, "fig_1_data_em.csv") %>%
  read_csv(col_types='iddddd') %>%
  mutate(x=u2, y=p1)

p <- ggplot(DF, aes(x, y)) +
       geom_raster(aes(fill=z), interpolate=T) +
       geom_contour(aes(z=z), colour="black", bins=15, alpha=0.25) +
       scale_fill_gradientn(colours=heat.colors(6)) +
       geom_line(data=em, color='black') +
       scale_x_continuous(TeX("$\\alpha$"), limits=c(0, 2), expand=c(0,0)) +
       scale_y_continuous(TeX("$\\pi_1$"), limits=c(0, 1), expand=c(0,0)) +
       geom_point(data=em %>% head(n=1), shape=1 , size=3) +
       geom_point(data=em %>% tail(n=1), shape=16, size=3) +
       geom_point(data=data.frame(x=1, y=0.5), shape='x', size=4, color='red') +
       my_theme() +
       theme(legend.position='none', axis.line=element_blank())

ggsave(sprintf("%s/%s", figures_folder, "fig_1.pdf"), p, width=4.5, height=2.5)


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

DFf <- sprintf("%s/%s", data_folder, "fig_2_data.csv") %>% read_csv(col_types='cdd')
fit1 <- DFf %>% filter(deg == 'alpha')
# read non-parametric degree log fit
DFnp <- DFf %>% filter(deg != 'alpha') %>%
  filter(coef<20) %>%
  mutate(deg=as.numeric(deg), stat=exp(coef), w=1/se)
# normalize by coefficient for degree 1
DFnp$stat <- DFnp$stat / DFnp[DFnp$deg==1, ]$stat
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

p <- ggplot(DF, aes(deg, stat, color=label)) +
       geom_point(shape=20, alpha=0.0) +
       geom_line(data=DF %>% filter(id=='ls'), size=0.5) +
       geom_line(data=DF %>% filter(id=='ldl'), size=0.5) +
       geom_point(data=DF %>% filter(id=='newman'), shape=20, alpha=0.7) +
       geom_point(data=DF %>% filter(id=='npl'), shape=20, alpha=0.7) +
       scale_x_log10("Degree", labels=trans_format('log10', math_format(10^.x)), breaks=c(10^0, 10^1, 10^2), expand=c(0,0), limits=c(0.9, 100)) +
       scale_y_log10("Relative probability", labels=trans_format('log10', math_format(10^.x)), breaks=c(10^0, 10^1, 10^2), expand=c(0,0), limits=c(0.9, 100)) +
       coord_cartesian(xlim=c(0.9, 100), ylim=c(0.9, 100)) +
       scale_color_manual(values=c("#E69F00", "#56B4E9", "#009E73", "#CC79A7"),
                          guide=guide_legend(override.aes=list(
                            linetype=c("blank", "blank", "solid", "solid"), shape=c(16, 16, NA, NA)))) +
       my_theme()

ggsave(sprintf("%s/%s", figures_folder, "fig_2.pdf"), p, width=4.5, height=2.5)



##
## Figure 3 - Power law fits on degree of (r,p) graphs
##

DF <- sprintf("%s/%s", data_folder, "fig_3_data.csv") %>%
  read_csv(col_types='cccccd') %>%
  group_by(type, r, p, type2) %>%
  summarize(mean_a=mean(alpha)) %>%
  ungroup() %>%
  mutate(
    # offset r slightly for plot
    r=as.numeric(r) + (as.numeric(p) - 0.5) / 50,
    Type=ifelse(type=='g', 'External / Growth', 'Internal / Densify'),
    Type2=ifelse(type2=='d', 'Directed', 'Undirected'),
    # compute p_hat
    p_hat=ifelse(type2=='d', (mean_a-2)/(mean_a-1), (mean_a-3)/(mean_a-1))
  )

p <- ggplot(DF, aes(x=r, y=mean_a, color=p)) +
       geom_line() +
       geom_point(show.legend=F) +
       scale_color_manual(values=c("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00")) + 
       scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
       scale_y_continuous(TeX("Estimate of $\\gamma$"), expand=c(0,0), limits=c(2.6, 5.7)) +
       geom_hline(yintercept=3, color='grey', linetype='dashed') +
       my_theme() +
       theme(legend.title=element_text(hjust=0.5))

ggsave(sprintf("%s/%s", figures_folder, "fig_3.pdf"), p, width=4.5, height=2.5)



##
## Figure 4 - Log-likelihood of misspecified models
##

DF <- sprintf("%s/%s", data_folder, "fig_4_data.csv") %>%
  read_csv(col_types='ccdd') %>%
  mutate(Model=ifelse(model=='p', 'Copy\nmodel', ifelse(model=='r', 'Local\nsearch', '2d')))

DF_max <- DF %>% group_by(data, Model) %>% filter(ll==max(ll))

# likelihood-ratio tests
lrt(DF_max$ll[4], DF_max$ll[3])
lrt(DF_max$ll[2], DF_max$ll[1])

p1 <- DF %>% filter(data=='r=0.50, p=1.00') %>%
        ggplot(aes(p, ll, color=Model)) +
          geom_line() +
          geom_point(data=DF_max %>% filter(data=='r=0.50, p=1.00'), show.legend=F) +
          scale_x_continuous(TeX("Class probability ($\\textit{p}$ or $\\textit{r}$)"), labels=c('0','0.25','0.50','0.75','1')) +
          scale_y_continuous("Log-likelihood", limits=c(-43000,-25000), labels=function(x) sprintf("%.0fk", x/1000)) +
          scale_color_manual(values=c("#E69F00", "#56B4E9")) +
          ggtitle(TeX("$\\textit{r}\ =\ 0.50\\;\\;\\textit{p}\ =\ 1.00$")) +
          my_theme() +
          theme(legend.position='none')

p2 <- DF %>% filter(data=='r=1.00, p=0.50') %>%
        ggplot(aes(p, ll, color=Model)) + geom_line() +
          geom_point(data=DF_max %>% filter(data=='r=1.00, p=0.50'), show.legend=F) +
          scale_x_continuous(TeX("Class probability ($\\textit{p}$ or $\\textit{r}$)"), labels=c('0','0.25','0.50','0.75','1')) +
          scale_y_continuous("Log-likelihood", limits=c(-43000, -25000), labels=function(x) sprintf("%.0fk", x/1000)) +
          scale_color_manual(values=c("#E69F00", "#56B4E9")) +
          ggtitle(TeX("$\\textit{r}\ =\ 1.00\\;\\;\\textit{p}\ =\ 0.50$")) +
          my_theme() +
          theme(legend.position=c(0.85, 0.85), axis.title.y=element_blank(), axis.text.y=element_blank())

pdf(sprintf("%s/%s", figures_folder, "fig_4.pdf"), width=6, height=3)
lay <- rbind(c(rep(1, 10), rep(2, 9)),
             c(rep(1, 10), rep(2, 9)))
grid.arrange(p1, p2, layout_matrix=lay)
dev.off()



##
## Figure 5 - Non-parametric estimates per model
##

DF <- rbind(
    sprintf("%s/%s", data_folder, "fig_5_flickr.csv") %>%
      read_csv(col_types='ddcc') %>%
      mutate(Model=paste0('3.', model), data='Flickr'),
    sprintf("%s/%s", data_folder, "fig_5_citations.csv") %>%
      read_csv(col_types='ddcc') %>%
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
    guides(colour=guide_legend(override.aes=list(alpha=1))) +
    scale_color_manual(values=c("#E69F00", "#56B4E9", "#009E73")) +
    ggtitle("Flickr") +
    my_theme() +
    theme(legend.position=c(0.89, 0.18))

p2 <- DF %>% filter(type=='point', data=='Citations') %>%
  ggplot(aes(deg, est, color=Model)) + 
    geom_line(alpha=0) +
    geom_line(data=data.frame(deg=1:100, est=1:100), color='black', linetype='dashed') +
    geom_point(shape=16, size=.8, show.legend=F) +
    geom_point(data=DF %>% filter(type=='point', deg==0.5, data=='Citations'), shape=16, size=1.2, show.legend=F) +
    geom_line(data=DF %>% filter(type=='line', data=='Citations'), alpha=0.6) +
    scale_x_log10('Degree', breaks=c(0.5, 1, 10, 100), labels=c(TeX("$\\,0^{\\;}$"), TeX("$10^0$"), TeX("$10^1$"), TeX("$10^2$"))) +
    scale_y_log10('Relative Probability', labels=trans_format('log10', math_format(10^.x)), limits=c(0.2, 250)) +
    guides(colour=guide_legend(override.aes=list(alpha=1))) +
    scale_color_manual(values=c("#E69F00", "#56B4E9")) +
    ggtitle("Citations") +
    my_theme() +
    theme(legend.position=c(0.89, 0.13), axis.title.y=element_blank(), axis.text.y=element_blank())

pdf(sprintf("%s/%s", figures_folder, "fig_5.pdf"), width=6, height=3)
lay <- rbind(c(rep(1, 10), rep(2, 9)),
             c(rep(1, 10), rep(2, 9)))
grid.arrange(p1, p2, layout_matrix=lay)
dev.off()
