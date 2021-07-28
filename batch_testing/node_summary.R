setwd('~/Desktop/batch_testing/')

library(tidyverse)

more_sim


more_sim = read_csv('more_sim.csv')
more_sim %>%
  filter(p == 0.03, `upper limit` == 1000000) %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  select(node:value) %>%
  group_by(node) %>%
  summarise(num = mean(value),
            std = sd(value))%>%
  view()

node_result_no %>%
  filter(p == 0.03, `upper limit` == 1000000) %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  select(node:value) %>%
  group_by(node) %>%
  summarise(num = mean(value),
            std = sd(value))%>%
  view()


node_result = read_csv('node_test.csv')
node_result %>% colnames()

node_result_no = read_csv('node_test_without_threshold.csv')


node_result %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  filter(p == 0.001) %>%
  mutate(`upper limit` = factor(`upper limit`))

node_batch_p_001 <- node_result %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  filter(p == 0.001) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = node, y = value, fill = upper_limit))+
    geom_boxplot() +
  facet_wrap(~node, scale = 'free') + 
  ylab(NULL) + 
  ggtitle('p = 0.001')

node_batch_p_001

node_result_no %>%
  filter(p == 0.03, `upper limit` == 1000000) %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  select(node:value) %>%
  group_by(node) %>%
  summarise(num = mean(value),
            std = sd(value))%>%
  view()



node_result %>%
  filter(p == 0.01) %>%
  select('+++':'--++-') %>%
  summarise_if(is.numeric, mean) %>%
  view()


node_batch_p_01 <- node_result %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  filter(p == 0.01) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = node, y = value, fill = upper_limit))+
  geom_boxplot() +
  facet_wrap(~node, scale = 'free') + 
  ylab(NULL) + 
  ggtitle('p = 0.01')

node_batch_p_01

node_batch_p_03 <- node_result %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  filter(p == 0.03) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = node, y = value, fill = upper_limit))+
  geom_boxplot() +
  facet_wrap(~node, scale = 'free') + 
  ylab(NULL) + 
  ggtitle('p = 0.03')

node_batch_p_03


node_batch_p_05 <- node_result %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  filter(p == 0.05) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = node, y = value, fill = upper_limit))+
  geom_boxplot() +
  facet_wrap(~node, scale = 'free') + 
  ylab(NULL) + 
  ggtitle('p = 0.05')

node_batch_p_05

node_batch_p_10 <- node_result %>%
  pivot_longer(cols = '+++':'other', names_to = 'node',
               values_to = 'value') %>%
  filter(p == 0.1) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = node, y = value, fill = upper_limit))+
  geom_boxplot() +
  facet_wrap(~node, scale = 'free') + 
  ylab(NULL) + 
  ggtitle('p = 0.1')

node_batch_p_10




batch_cum <- read_csv('batch_cum.csv')
batch_cum

batch_cum_001 <- batch_cum %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
        values_to = 'value') %>%
  filter(p == 0.001) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = upper_limit, y = value, fill = upper_limit))+ 
  geom_boxplot() + 
  facet_wrap(~stage, scale = 'free') +
  ggtitle('p = 0.001')

batch_cum_001

batch_cum_line_001 <-batch_cum %>%
  group_by(p, `upper limit`) %>%
  summarise_at(vars(stage_1:stage_5), mean) %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p ==  0.001) %>%
  mutate(upper_limit = factor(`upper limit`)) %>%
  ggplot(aes(x = stage, y = value, group = upper_limit, color = upper_limit)) +
    geom_point(stat = 'summary', fun = sum) + 
  stat_summary(fun = sum, geom = 'line') + 
  xlab(NULL) + 
  ylab('Cumulative Batch Test Consumption') + 
  ggtitle('p = 0.001')

batch_cum_line_001  



batch_cum_01 <- batch_cum %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p == 0.01) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = upper_limit, y = value, fill = upper_limit))+ 
  geom_boxplot() + 
  facet_wrap(~stage, scale = 'free')+
  ggtitle('p = 0.01')

batch_cum_01

batch_cum_line_01 <-batch_cum %>%
  group_by(p, `upper limit`) %>%
  summarise_at(vars(stage_1:stage_5), mean) %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p ==  0.01) %>%
  mutate(upper_limit = factor(`upper limit`)) %>%
  ggplot(aes(x = stage, y = value, group = upper_limit, color = upper_limit)) +
  geom_point(stat = 'summary', fun = sum) + 
  stat_summary(fun = sum, geom = 'line') + 
  xlab(NULL) + 
  ylab('Cumulative Batch Test Consumption') + 
  ggtitle('p = 0.01')

batch_cum_line_01  


batch_cum_03 <- batch_cum %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p == 0.03) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = upper_limit, y = value, fill = upper_limit))+ 
  geom_boxplot() + 
  facet_wrap(~stage, scale = 'free')+
  ggtitle('p = 0.03')

batch_cum_03

batch_cum_line_03 <-batch_cum %>%
  group_by(p, `upper limit`) %>%
  summarise_at(vars(stage_1:stage_5), mean) %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p ==  0.03) %>%
  mutate(upper_limit = factor(`upper limit`)) %>%
  ggplot(aes(x = stage, y = value, group = upper_limit, color = upper_limit)) +
  geom_point(stat = 'summary', fun = sum) + 
  stat_summary(fun = sum, geom = 'line') + 
  xlab(NULL) + 
  ylab('Cumulative Batch Test Consumption') + 
  ggtitle('p = 0.03')

batch_cum_line_03 



batch_cum_05 <- batch_cum %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p == 0.05) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = upper_limit, y = value, fill = upper_limit))+ 
  geom_boxplot() + 
  facet_wrap(~stage, scale = 'free')+
  ggtitle('p = 0.05')

batch_cum_05

batch_cum_line_05 <-batch_cum %>%
  group_by(p, `upper limit`) %>%
  summarise_at(vars(stage_1:stage_5), mean) %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p ==  0.05) %>%
  mutate(upper_limit = factor(`upper limit`)) %>%
  ggplot(aes(x = stage, y = value, group = upper_limit, color = upper_limit)) +
  geom_point(stat = 'summary', fun = sum) + 
  stat_summary(fun = sum, geom = 'line') + 
  xlab(NULL) + 
  ylab('Cumulative Batch Test Consumption') + 
  ggtitle('p = 0.05')

batch_cum_line_05  



batch_cum_10 <- batch_cum %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p == 0.1) %>%
  mutate(`upper_limit` = factor(`upper limit`)) %>%
  ggplot(aes(x = upper_limit, y = value, fill = upper_limit))+ 
  geom_boxplot() + 
  facet_wrap(~stage, scale = 'free')+
  ggtitle('p = 0.1')

batch_cum_10

batch_cum_line_10 <-batch_cum %>%
  group_by(p, `upper limit`) %>%
  summarise_at(vars(stage_1:stage_5), mean) %>%
  pivot_longer(cols = stage_1:stage_5, names_to = 'stage',
               values_to = 'value') %>%
  filter(p ==  0.1) %>%
  mutate(upper_limit = factor(`upper limit`)) %>%
  ggplot(aes(x = stage, y = value, group = upper_limit, color = upper_limit)) +
  geom_point(stat = 'summary', fun = sum) + 
  stat_summary(fun = sum, geom = 'line') + 
  xlab(NULL) + 
  ylab('Cumulative Batch Test Consumption') + 
  ggtitle('p = 0.1')

batch_cum_line_10  



##

table7_f <- read_csv('table7_f.csv')
table7_f <- table7_f %>%
  mutate(batch_limit = 100000)
table7_f_limit_32 <- read_csv('table7_f_limit_32.csv') %>%
  mutate(batch_limit = 32)
table7_f_limit_64 <- read_csv('table7_f_limit_64.csv') %>%
  mutate(batch_limit = 64)


table_limit_vs_no <- bind_rows(table7_f, table7_f_limit_32,
                               table7_f_limit_64)
table_limit_vs_no %>%
  select(Infection_rate, Test_consum, Batch_consum, batch_limit) %>%
  mutate(batch_limit = factor(batch_limit)) %>%
  ggplot(aes(batch_limit, Test_consum))+
  geom_col() + 
  facet_wrap(~Infection_rate, scales = 'free_y')+
  ggtitle('Test Consumption')




table_limit_vs_no %>%
  select(Infection_rate, Test_consum, Batch_consum, batch_limit) %>%
  mutate(batch_limit = factor(batch_limit)) %>%
  ggplot(aes(batch_limit, Batch_consum))+
  geom_col() + 
  facet_wrap(~Infection_rate, scales = 'free_y')+
  ggtitle('Batch Consumption')


result_batch_limit <- read_csv('result_batch_limit.csv')


result_batch_limit <- result_batch_limit %>%
  select(-X1) %>%
  mutate(Infection_rate = as.factor(Infection_rate),
         Batch_upper_limit = as.factor(Batch_upper_limit))


result_batch_limit %>%
  pivot_longer(
    cols = c(Batch_consum, Ind_consum),
    names_to = 'Type',
    values_to = 'value'
  ) %>%
  ggplot(aes(Batch_upper_limit, value, fill = Type)) +
  geom_col() + 
  facet_wrap(~Infection_rate, scales = 'free_y' )+
  expand_limits(y = 0) + 
  #scale_y_continuous(expand = c(0, 0), limits = c(0, NA)) +
  labs(y = NULL, x = 'Batch Size Upper Limits') + 
  theme_bw()


result_batch_limit %>%
  pivot_longer()
  ggplot(aes(Batch_upper_limit, Ind_consum, Batch_consum))+
  geom_col() + 
    
  facet_wrap(~Infection_rate)
  
1/.42
