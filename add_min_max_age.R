library(tidyverse)

data = read_csv("bat_speed_23-25.csv") 

data_clean = data %>% 
  drop_na() %>% 
  filter(between(player_age, 20, 40))

colnames(data_clean)[1] = "player_name" 

min_max_age = data_clean %>% 
  group_by(player_id) %>% 
  summarize(
    max_age = max(player_age), 
    min_age = min(player_age)
  ) 

data_clean = data_clean %>% 
  left_join(
    min_max_age, by = "player_id"
  ) 

write_csv(data_clean, "bat_speed_df.csv") 
