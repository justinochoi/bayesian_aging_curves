library(tidyverse)
library(mgcv)

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

players = unique(data_clean$player_id) 
year_grid = expand.grid(year = c(2023,2024,2025), player_id = players)
year_grid = year_grid %>% 
  left_join(data_clean, by = c("year", "player_id"))

year_grid = year_grid %>% 
  group_by(player_id) %>% 
  mutate(
    player_age = case_when(
      is.na(player_age) & !is.na(lead(player_age)) ~ lead(player_age) - 1, 
      is.na(player_age) & !is.na(lag(player_age)) ~ lag(player_age) + 1, 
      is.na(player_age) & is.na(lead(player_age)) & !is.na(lead(player_age, n=2)) ~ lead(player_age, n=2) - 2, 
      is.na(player_age) & is.na(lag(player_age)) & !is.na(lag(player_age, n=2)) ~ lag(player_age, n=2) + 2, 
      .default = player_age
    )
  ) %>% 
  ungroup() 

year_grid = year_grid %>% 
  mutate(
    observed = if_else(is.na(avg_swing_speed), 0, 1)
  )

write_csv(year_grid, "bat_speed_heckman.csv") 


# the effect of age on observing a player 
age_model = gam(observed ~ s(player_age), data = year_grid, family = "binomial")
summary(age_model) 
plot(age_model) 

# less likely to observe younger AND older players 
# the effect of age on observing is like an upside-down U

