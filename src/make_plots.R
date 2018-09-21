# Run with Rscript make_plots.R

library(dplyr)
library(ggplot2)
library(latex2exp)
require(tidyr)
require(readr)

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
## Figure 5.1
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
    my_theme(11) -> p51

ggsave('../results/fig_5_1.pdf', p51, width=6, height=3)


DF %>%
  filter(Model != 'p-single') %>%
  mutate(Model=ifelse(Model=='p-mixed', 'PA', 'PA-FoF')) %>%
  ggplot(aes(r, LL, color=Model)) +
    geom_line() +
    facet_grid(Type ~ P, scales='free_y') +
    scale_color_brewer(palette='Set1') + 
    scale_x_continuous("r", breaks=seq(0, 1, 0.25), labels=c('0','0.25','0.50','0.75','1')) +
    scale_y_continuous("Log-likelihood") +
    my_theme(11) -> p52

ggsave('../results/fig_5_2.pdf', p52, width=9, height=3.5)

