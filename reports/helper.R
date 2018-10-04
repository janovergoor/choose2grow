suppressPackageStartupMessages(library(ggplot2))
library(mlogit)
library(scales)
library(Rmisc)
library(stringr)
library(tidyr)
library(dplyr)
library(readr)

# cleaner theme
my_theme <- function(base_size=10) {
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
      legend.title=element_blank(),
      legend.background=element_rect(fill='transparent')
    )
}

# read plfit from external source
source("http://tuvalu.santafe.edu/~aaronc/powerlaws/plfit.r")

# CDF of power law distribution
cdf <- Vectorize(function(x, a, xmin){
  (x/xmin)^(-a+1)
})

# approximate Zeta function
my_zeta <- Vectorize(function(a, xmin) {
  sum((0:100000 + xmin)^(-a))
})

# plot a inverse CDF of a degree distribution with a power law fit overlaid
plot_powerlaw_cdf <- function(X, title, xlab, ylab) {
  fit <- plfit(X, "range", seq(1.001,4,0.01))
  print(sprintf("plfit:  alpha=%.3f  xmin=%d", fit$alpha, fit$xmin))
  DF <- data.frame(x=1:max(X)) %>%
    left_join(data.frame(x=X) %>% group_by(x) %>% summarize(n=n()), by='x') %>%
    mutate(n=ifelse(is.na(n), 0, n)) %>%
    mutate(p=(sum(n)-cumsum(n))/sum(n))
  ggplot(DF, aes(x, p)) + geom_point() +
    ggtitle(title) +
    scale_x_log10(xlab, breaks=c(1,10,100,1000), labels=trans_format('log10', math_format(10^.x))) +
    scale_y_log10(ylab, breaks=c(10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6), labels=trans_format('log10', math_format(10^.x))) +
    geom_line(data=data.frame(x=fit$xmin:max(X)) %>% mutate(p=cdf(x, fit$alpha, fit$xmin)*DF$p[fit$xmin]), color='red') +
    my_theme()
}

# compute the accuracy of a model on new data
acc <- function(f, data) {
  # compute predictions
  P = predict(f, newdata=data) %>% as.data.frame()
  inner_join(
    # actuals
    data %>% filter(y) %>% select(paper_id, correct=alt_id),
    # predicted
    P %>% mutate(paper_id=rownames(P)) %>% gather(choice, score, -paper_id) %>% group_by(paper_id) %>% filter(score==max(score)),
    by='paper_id'
  ) %>% ungroup() %>%
    summarize(acc=mean(choice==correct)) %>% .$acc %>% as.numeric()
}
