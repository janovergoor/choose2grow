# Run with `Rscript paper2/make_plots.R`

source('../helper.R')
base_size = 10
data_folder = '../../data/paper2_plot_data'
figures_folder = '../../figures/paper2/'


##
## Figure 2 - Time complexity
##

p2 <- sprintf("%s/%s", data_folder, "fig_2_data.csv") %>%
      read_csv() %>%
      group_by(n, s) %>%
      summarize(time=log(mean(time))) %>%
      ungroup() %>%
      rbind(data.frame(n=100000, s=3000, time=9.0)) %>%
      filter(n>=30, s>=3) %>%
      ggplot(aes(n, s, z=time)) + 
        geom_raster(aes(fill=time), interpolate=T) +
        stat_contour(bins=12, color='black', alpha=0.5) +
        scale_x_log10("Number of data points (n)", expand=c(0,0), labels=trans_format('log10', math_format(10^.x))) +
        scale_y_log10("Number of samples (s)", expand=c(0,0), labels=trans_format('log10', math_format(10^.x))) +
        coord_cartesian(xlim=c(30, 100000), ylim=c(3, 1000)) +
        scale_fill_gradientn(name="Runtime\n (sec)", colours=rev(heat.colors(10)),
                             breaks=log(c(10^1, 10^3)),
                             labels=c(expression(10^1), expression(10^3))) +
        my_theme() +
        theme(legend.title=element_text())

ggsave(sprintf("%s/%s", figures_folder, "fig_2.pdf"), p2, width=5, height=3, dpi=500)



##
## Figure 3 - Synthetic experiment with variable n and s
##

# Features
#   0 : log(alter degree)
#   2 : log(i -> j) 'repetition'
#   1 : log(j -> i) 'reciprocity'
#   3 : log(paths)
#   4 : I(alter deg > 0)
#   5 : I(i -> j > 0)
#   6 : I(j -> i > 0)
#   7 : I(paths > 0)

d3ab <- sprintf("%s/%s", data_folder, "fig_3ab_data.csv") %>%
  read_csv() %>%
  gather(var, val, -num_nodes, -graph_id, -sampling, -s, -data_size, -run_id) %>%
  separate(var, into=c('what', 'var'), sep='_') %>%
  mutate(sampling=factor(sampling, levels=c("uniform","importance"), labels=c("Uniform","Importance")))

p3a <- d3ab %>%
  # w2 = coefficient for reciprocity
  filter(what=='w', var=='2', data_size %% 500 == 0, s==24) %>% 
  group_by(data_size, s, sampling, var) %>%
  summarize(q05=quantile(val, 0.05, na.rm=T),
            q25=quantile(val, 0.25, na.rm=T),
            med=quantile(val, 0.50, na.rm=T),
            q75=quantile(val, 0.75, na.rm=T),
            q95=quantile(val, 0.95, na.rm=T)
  ) %>%
  ungroup() %>%
  mutate(x=data_size + 90*(as.numeric(sampling)-1)) %>%
  ggplot(aes(x=x, color=sampling)) +
    geom_point(aes(y=med), shape=20) +
    geom_segment(aes(xend=x, y=q05, yend=q95)) +
    geom_segment(aes(xend=x, y=q25, yend=q75), size=1.1) + 
    scale_x_continuous("Number of data points (n)", breaks=seq(1000, 6000, 1000)) +
    scale_y_continuous("Estimate") + coord_cartesian(ylim=c(0.5, 5)) +
    scale_color_manual(values=c("#E69F00", "#56B4E9"),
                       labels=c("Uniform", "Importance"),
                       guide=guide_legend(override.aes=list(linetype=c("blank", "blank"), shape=c(16, 16)))) +
    ggtitle("s Constant") +
    my_theme(base_size) +
    theme(legend.position='none')

p3b <- inner_join(
    # w2 = coefficient for reciprocity
    d3ab %>% 
      filter(data_size %% 500 == 0, what=='w', var=='2', s==24) %>%
      select(sampling, run_id, data_size, val, coef=val),
    d3ab %>%
      filter(data_size %% 500 == 0, what=='se'  , var=='2', s==24) %>%
      select(sampling, run_id, data_size, val, se=val),
    by=c("sampling","data_size","run_id")
  ) %>%
  mutate(mse=(coef - 2) ** 2 + se ** 2) %>%
  group_by(data_size, sampling) %>%
  summarize(q50=quantile(mse, 0.50, na.rm=T),
            q05=quantile(mse, 0.25, na.rm=T),
            q95=quantile(mse, 0.75, na.rm=T)
  ) %>%
  ungroup() %>%
  ggplot(aes(x=data_size, color=sampling)) +
    geom_segment(aes(xend=data_size, y=q05, yend=q95)) +
    geom_point(aes(y=q50), shape=20) +
    scale_color_manual(values=c("#E69F00", "#56B4E9"),
                       labels=c("Uniform", "Importance"),
                       guide=guide_legend(override.aes=list(linetype=c("blank", "blank"), shape=c(16, 16)))) +
    scale_x_continuous("Number of data points (n)", breaks=seq(1000, 6000, 1000)) +
    scale_y_log10("MSE") +
    ggtitle("s Constant") +
    my_theme(base_size=9) +
    theme(legend.pos='none')


d3c <- sprintf("%s/%s", data_folder, "fig_3c_data.csv") %>%
  read_csv() %>%
  gather(var, val, -num_nodes, -graph_id, -sampling, -s, -data_size, -run_id) %>%
  separate(var, into=c('what', 'var'), sep='_') %>%
  # w2 = coefficient for reciprocity
  filter(var=='2') %>%
  select(sampling, s, what, val, run_id) %>%
  mutate(sampling=factor(sampling,
                         levels=c("uniform","stratified","importance"),
                         labels=c("Uniform","Stratified","Importance")))

p3c <- d3c %>%
  filter(val < 5, sampling != 'All', sampling != 'Importance') %>% 
  spread(what, val) %>%
  mutate(mse=(w - 2) ** 2 + se ** 2) %>%
  group_by(sampling, s) %>%
  summarize(
    q50=quantile(mse, 0.50, na.rm=T),
    q05=quantile(mse, 0.25, na.rm=T), 
    q95=quantile(mse, 0.75, na.rm=T)
  ) %>% 
  ungroup() %>%
  mutate(x=s) %>%
  ggplot(aes(x=x, color=sampling)) +   
    geom_point(aes(y=q50), shape=20) +
    geom_segment(aes(xend=x, y=q05, yend=q95)) +
    scale_color_manual(values=c("#E69F00", "#56B4E9"),
                       labels=c("Uniform", "Importance"),
                       guide=guide_legend(override.aes=list(linetype=c("blank", "blank"), shape=c(16, 16)))) +
    scale_x_log10("Number of samples (s)", breaks=d3c %>% mutate(s=as.numeric(s)) %>% distinct(s) %>% .$s) +
    scale_y_log10("MSE") +
    ggtitle("n Constant") +
    my_theme(base_size=9) +
    theme(legend.pos='none')


d3d <- sprintf("%s/%s", data_folder, "fig_3d_data.csv") %>%
  read_csv() %>%
  filter(s < 770) %>%
  gather(var, val, -num_nodes, -graph_id, -sampling, -s, -data_size, -run_id) %>%
  separate(var, into=c('what', 'var'), sep='_') %>%
  # w2 = coefficient for reciprocity
  filter(var=='2') %>%
  select(sampling, s, what, val, run_id) %>%
  mutate(sampling=factor(sampling, 
                         levels=c("uniform","stratified","importance"),
                         labels=c("Uniform","Stratified","Importance")))

p3d <- d3d %>%
  filter(val < 5, sampling != 'All', sampling != 'Importance') %>%
  spread(what, val) %>%
  mutate(mse=(w - 2) ** 2 + se ** 2) %>%
  group_by(sampling, s) %>%
  summarize(
    q50=quantile(mse, 0.50, na.rm=T),
    q05=quantile(mse, 0.25, na.rm=T),
    q95=quantile(mse, 0.75, na.rm=T)
  ) %>%
  ungroup() %>%
  mutate(x=s) %>%
  ggplot(aes(x=x, color=sampling)) +  
    geom_segment(aes(xend=x, y=q05, yend=q95)) +
    geom_point(aes(y=q50), shape=20) + 
    scale_color_manual(values=c("#E69F00", "#56B4E9"),
                       labels=c("Uniform", "Importance"),
                       guide=guide_legend(override.aes=list(linetype=c("blank", "blank"), shape=c(16, 16)))) +
    scale_x_log10("Number of samples (s)", breaks=d3d %>% mutate(s=as.numeric(s)) %>% distinct(s) %>% .$s) +
    scale_y_log10("MSE", labels=c('.003','.010','.030','.100','.300'), breaks=c(.003,.010,.030,.100,.300)) + 
    ggtitle("n*s Constant") +
    my_theme(base_size) +
    theme(#axis.title.y=element_blank(),
          legend.position=c(0.85, 0.14),
          legend.text=element_text(size=rel(0.7)),
          legend.key.size = unit(0.3, "cm"))

ggsave(sprintf("%s/%s", figures_folder, "fig_3.pdf"),
       Rmisc::multiplot(p3a, p3c, p3b, p3d, cols=2),
       width=5, height=4.2)




##
## Figure 4 - Synthetic experiment comparing CL and ML estimates
##

d4ab <- sprintf("%s/%s", data_folder, "fig_4ab_data.csv") %>%
  read_csv() %>%
  gather(var, val, -num_nodes, -graph_id, -sampling, -s, -data_size, -run_id) %>%
  separate(var, into=c('what', 'var'), sep='_') %>%
  mutate(sampling=factor(sampling,
                         levels=c("uniform","stratified"),
                         labels=c("Uniform","Stratified")))

p4ab <- d4ab %>%
  # w0 = coefficient for log degree
  # w6 = coefficient for I(j -> i > 0)s
  filter(what=='w', var %in% c("0", "6")) %>% 
  mutate(var=ifelse(var=='0', 'log Degree' , "Reciprocity (ind)")) %>%
  group_by(sampling, var, s) %>%
  summarize(q05=quantile(val, 0.05, na.rm=T),
            q25=quantile(val, 0.25, na.rm=T),
            med=quantile(val, 0.50, na.rm=T),
            q75=quantile(val, 0.75, na.rm=T),
            q95=quantile(val, 0.95, na.rm=T)
  ) %>%
  ungroup() %>%
  mutate(x=s * (1 + 0.07 * (as.numeric(as.factor(sampling)) - 1))) %>% 
  ggplot(aes(x=x, color=sampling)) +  
    geom_point(aes(y=med), shape=20) +
    geom_segment(aes(xend=x, y=q05, yend=q95)) +
    geom_segment(aes(xend=x, y=q25, yend=q75), size=1.1) +
    geom_hline(data=data.frame(var=c('log Degree',"Reciprocity (ind)"), val=c(.5, 1)), aes(yintercept=val), color='lightgrey') + 
    geom_hline(data=data.frame(var=c('log Degree',"Reciprocity (ind)"), val=c(0, 0)), aes(yintercept=val), color='white') + 
    geom_hline(data=data.frame(var=c('log Degree',"Reciprocity (ind)"), val=c(1, 3)), aes(yintercept=val), color='white') + 
    scale_color_manual(values=c("#E69F00", "#56B4E9"),
                       labels=c("Uniform", "Importance"),
                       guide=guide_legend(override.aes=list(linetype=c("blank", "blank"), shape=c(16, 16)))) +
    scale_x_log10("s", breaks=d4ab %>% mutate(s=as.numeric(s)) %>% distinct(s) %>% .$s) +
    scale_y_continuous("CL Estimates", labels=function(x) sprintf("%.2f", x)) +
    facet_wrap(~var, scales='free_y', ncol=2) +
    my_theme(base_size) +
    theme(legend.position='none',
          axis.title.x=element_blank(),
          strip.background=element_blank())


d4cd <- sprintf("%s/%s", data_folder, "fig_4cd_data.csv") %>%
  read_csv() %>%
  gather(var, val, -num_nodes, -graph_id, -sampling, -s, -data_size, -run_id) %>%
  separate(var, into=c('what', 'mode', 'var'), sep='_') %>%
  mutate(sampling=factor(sampling,
         levels=c("uniform","stratified","importance"),
         labels=c("Uniform","Stratified","Importance")))

p4cd <- d4cd %>%
  # w0 = coefficient for log degree
  # w6 = coefficient for I(j -> i > 0)s
  filter(what=='w', var %in% c("0", "6"), sampling != 'Importance', mode=='local') %>% 
  mutate(var=ifelse(var=='0', 'log Degree' , "Reciprocity (ind)")) %>%
  group_by(sampling, var, s) %>%
  summarize(q05=quantile(val, 0.05, na.rm=T),
            q25=quantile(val, 0.25, na.rm=T),
            med=quantile(val, 0.50, na.rm=T),
            q75=quantile(val, 0.75, na.rm=T),
            q95=quantile(val, 0.95, na.rm=T)
  ) %>%
  ungroup() %>%
  mutate(x=s * (1 + 0.07 * (as.numeric(as.factor(sampling)) - 1))) %>% 
  ggplot(aes(x=x, color=sampling)) +  
    geom_point(aes(y=med), shape=20) +
    geom_segment(aes(xend=x, y=q05, yend=q95)) +
    geom_segment(aes(xend=x, y=q25, yend=q75), size=1.1) +
    geom_hline(data=data.frame(var=c('log Degree',"Reciprocity (ind)"), val=c(.5, 1)), aes(yintercept=val), color='lightgrey') + 
    geom_hline(data=data.frame(var=c('log Degree',"Reciprocity (ind)"), val=c(0, 0)), aes(yintercept=val), color='white') + 
    geom_hline(data=data.frame(var=c('log Degree',"Reciprocity (ind)"), val=c(1, 3)), aes(yintercept=val), color='white') + 
    geom_point(aes(y=med), shape=20) +
    geom_segment(aes(xend=x, y=q05, yend=q95)) +
    geom_segment(aes(xend=x, y=q25, yend=q75), size=1.1) +
    scale_color_manual(values=c("#E69F00", "#56B4E9"),
                       labels=c("Uniform", "Importance"),
                       guide=guide_legend(override.aes=list(linetype=c("blank", "blank"), shape=c(16, 16)))) +
    scale_x_log10("s", breaks=d4cd %>% mutate(s=as.numeric(s)) %>% distinct(s) %>% .$s) +
    scale_y_continuous("Demixed ML Estimates", labels=function(x) sprintf("%.2f", x)) +
    facet_wrap(~var, scales='free_y', ncol=2) +
    my_theme(base_size) +
    theme(legend.position=c(0.92, 0.14),
          legend.text=element_text(size=rel(0.8)),
          legend.key.size = unit(0.4, "cm"),
          strip.background = element_blank(), 
          strip.text.x = element_blank())

ggsave(sprintf("%s/%s", figures_folder, "fig_4.pdf"),
       Rmisc::multiplot(p4ab, p4cd),
       width=5, height=3.8)


##
## Figure 5 - Venmo demographics
## 

p5a <- sprintf("%s/%s", data_folder, "fig_5a_data.csv") %>%
  read_csv() %>%
  mutate(week = as.Date(cut(date, "week"))) %>%
  filter(week < as.Date("2018-07-16")) %>%
  group_by(week) %>%
  summarize(n=sum(txn_count)) %>%
  ungroup() %>% 
  ggplot(aes(week, n)) +
    geom_line(size=0.5) +
    xlab("Week") +
    scale_y_continuous("Number of transactions", expand=c(0,0), breaks=c(2e+06, 4e+06, 6e+06), labels=c("1M","2M","3M")) +
    my_theme()

p5b <- sprintf("%s/%s", data_folder, "fig_5b_data.csv") %>%
  read_csv() %>%
  group_by(Type) %>%
  mutate(p=1-(cumsum(n)/sum(n))) %>% #mutate(p=n/sum(n)) %>%
  ggplot(aes(Degree, p, color=Type)) +
    geom_point(alpha=0.7) +
    scale_x_log10("Degree (log)", labels=trans_format('log10', math_format(10^.x))) +
    scale_y_log10("CCDF", labels=trans_format('log10',  math_format(10^.x))) +
    scale_colour_manual(values=c("#E69F00", "#56B4E9")) +
    my_theme() +
    theme(legend.position=c(0.81, 0.89))

ggsave(sprintf("%s/%s", figures_folder, "fig_5.pdf"), 
       # hack for uneven width
       grid.arrange(p5a, p5b, width=6.5, height=3, layout_matrix=rbind(c(rep(1, 8), rep(2, 10)), c(rep(1, 8), rep(2, 10)))))



##
## Figure 6 - Non-Parametric estimates for Venmo
##

DF <- sprintf("%s/%s", data_folder, "fig_6_data.csv") %>% read_csv()

DF <- DF %>%
  left_join(DF %>% filter(deg==1) %>% select(coef1=coef, Mode), by='Mode') %>%
  mutate(rel_coef=coef/coef1, coef_exp=exp(coef), rel_coef_exp=exp(coef)/exp(coef1))

p6 <- DF %>%
  filter(deg != 0) %>%
  ggplot(aes(deg, rel_coef_exp, color=Mode)) +
    geom_point(shape=16, size=1, alpha=0.8) +
    geom_point(data=DF %>% filter(deg==0) %>% mutate(deg=0.5), shape=16, size=1.3) +
    scale_x_log10('In-degree'           , breaks=c(0.5, 1, 10, 100, 300), labels=c(0, 1, 10, 100, 300)) +
    scale_y_log10('Relative Probability', breaks=10^c(-0.5, 0, 0.5, 1, 1.5, 2), expand=c(0, 0), labels=trans_format('log10', math_format(10^.x)), limits=c( 10^(-0.5),  10^(2))) +
    scale_color_manual(values=c("#E69F00", "#56B4E9", "#009E73")) +
    my_theme() +
    theme(legend.position=c(0.87, 0.13))

ggsave(sprintf("%s/%s", figures_folder, "fig_6.pdf"), p6, width=4, height=2.6)
