suppressPackageStartupMessages(library(ggplot2))
library(mlogit)
library(scales)
library(pROC)
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

# compute Jackson's r on a degree distribution
# based on code by Eduardo Muggenburg
r_jackson <- function(deg_dist, m, tol=0.00000001 , r0=1.3, r1=1.7, max_iter=500){
  LHS <- log(1 - deg_dist + 0.000001)
  delta <- abs(r0 - r1)
  degrees <- 1:length(deg_dist)
  k <- 1
  while(delta > tol & k < max_iter){
    RHS <- log(degrees + r0 * m)
    f <- lm(LHS ~ 0 + RHS)
    r_tmp <- summary(f)$coefficients[1,1]
    r1 <- (-1*r_tmp)-1
    delta <- abs(r0 - r1)
    r0 <- r1
    k <- k + 1
  }
  r1
}

# compute the accuracy of a model on new data
acc <- function(f, data) {
  # compute predictions
  P <- predict(f, newdata=data) %>% as.data.frame()
  dp <- P %>% mutate(choice_id=rownames(P)) %>% gather(choice, score, -choice_id)
  inner_join(
    # actuals
    data %>% filter(y) %>% select(choice_id, correct=alt_id) %>% mutate(choice_id=as.character(choice_id)),
    # predicted
    dp %>%
      # sort by score and random to break ties randomly
      mutate(r=runif(nrow(dp))) %>% group_by(choice_id) %>% arrange(-score, r) %>%
      # take highest score
      mutate(n=row_number()) %>% filter(n==1),
    by='choice_id'
  ) %>% ungroup() %>%
    summarize(acc=mean(choice==correct)) %>% .$acc %>% as.numeric()
}

# compute the AUC of a model on new data
auc <- function(f, data) {
  # compute predictions
  P <- predict(f, newdata=data) %>% as.data.frame()
  dp <- P %>% mutate(choice_id=rownames(P)) %>% gather(choice, score, -choice_id)
  inner_join(
    # actuals
    data %>% filter(y) %>% select(choice_id, correct=alt_id) %>% mutate(choice_id=as.character(choice_id)),
    # predicted
    dp %>%
      # sort by score and random to break ties randomly
      mutate(r=runif(nrow(dp))) %>% group_by(choice_id) %>% arrange(-score, r) %>%
      # take highest score
      mutate(n=row_number()) %>% filter(n==1),
    by='choice_id'
  ) %>% ungroup() -> x
  # shuffle classes to be able to do multi-class AUC
  x$correct2 <- sample(1:25, nrow(x), replace=T)
  x$choice2 <- ifelse(x$choice==1, x$correct2, x$choice) # 1/25 chance of wrongly getting the 'right' answer
  #x$choice2 <- ifelse(x$choice!=1 & x$correct2==x$choice2, , x$choice2)
  as.numeric(multiclass.roc(as.numeric(x$correct2), as.numeric(x$choice2), quiet=T)$auc)
}