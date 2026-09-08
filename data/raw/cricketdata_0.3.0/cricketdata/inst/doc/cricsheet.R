## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  echo = TRUE
)

## ----setup--------------------------------------------------------------------
library(cricketdata)
library(readr)
library(dplyr)
library(stringr)
library(showtext)
library(ggplot2)
library(gghighlight)
library(ggtext)
library(patchwork)

## ----getdata, eval=FALSE, echo=FALSE------------------------------------------
# # Avoid downloading the data when the package is checked by CRAN.
# # This only needs to be run once to store the data locally
# wbbl_bbb <- fetch_cricsheet(competition = "wbbl", gender = "female")
# wbbl_match_info <- fetch_cricsheet(competition = "wbbl", type = "match", gender = "female")
# saveRDS(wbbl_bbb, "inst/extdata/wbbl_bbb.rds")
# saveRDS(wbbl_match_info, "inst/extdata/wbbl_match_info.rds")

## ----loaddata, include=FALSE--------------------------------------------------
wbbl_bbb <- readRDS("../inst/extdata/wbbl_bbb.rds")
wbbl_match_info <- readRDS("../inst/extdata/wbbl_match_info.rds")

## -----------------------------------------------------------------------------
# Data from 2015/16 (WBBL01) excluded due to only having 10 matches worth of data
# in the Cricsheet spreadsheet.
# Data from 2021/22 and later (WBBL07) excluded as incomplete at time of article
wbbl_bbb_tidy <- wbbl_bbb |>
  filter(!season %in% c("2015/16", "2021/22")) |>
  filter(start_date < "2021-11-07")

## -----------------------------------------------------------------------------
# Alyssa Healy compared to all players who have batted in 3+ innings in a season.
batting_per_season <- wbbl_bbb_tidy |>
  group_by(season, striker) |>
  summarise(
    innings_total = length(unique(match_id)),
    runs_off_bat_total = sum(runs_off_bat),
    balls_faced_total = length(ball),
    .groups = "keep"
  ) |>
  mutate(
    runs_per_innings_avg = round(runs_off_bat_total / innings_total, 1),
    strike_rate = round(runs_off_bat_total / balls_faced_total * 100, 1)
  ) |>
  filter(innings_total > 2) |>
  mutate(is_healy = (striker == "AJ Healy")) |>
  ungroup()

## ----warning=FALSE, out.width="100%"------------------------------------------
# Import fonts from Google Fonts
font_add_google("Roboto Condensed", "roboto_con")
font_add_google("Staatliches", "staat")
showtext_auto()

# Build plot
batting_per_season |>
  ggplot(aes(
    x = season, y = runs_per_innings_avg,
    group = striker, colour = is_healy
  )) +
  geom_line(linewidth = 2, colour = "#F80F61FF") +
  gghighlight(is_healy,
    label_key = striker,
    label_params = aes(
      size = 6, force_pull = 0.1, nudge_y = 10, label.size = 1,
      family = "roboto_con", label.padding = 0.5,
      fill = "#19232FFF",
      colour = "#F80F61FF"
    ),
    unhighlighted_params = list(size = 1, color = "#187999FF")
  ) +
  labs(
    title = "WBBL: Average runs scored per innings (3+ innings)",
    x = NULL, y = NULL,
    caption = "**Source:** Cricsheet.org // **Plot:** @jacquietran"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 18, family = "roboto_con", colour = "#FFFFFF"),
    plot.title = element_text(family = "staat", margin = margin(0, 0, 15, 0)),
    plot.caption = element_markdown(size = NULL, margin = margin(15, 0, 0, 0)),
    axis.text = element_text(colour = "#FFFFFF"),
    legend.position = "none",
    panel.grid.major = element_line(linetype = "dashed"),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(
      fill = "#171F2AFF", colour = NA
    ),
    panel.spacing = unit(2, "lines"),
    plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
  )

## -----------------------------------------------------------------------------
# Create new variable for ball number in each over
ball_number_faced <- wbbl_bbb_tidy |>
  mutate(ball_num_in_over = sub(".*\\.", "", ball))

# Summarise number of balls faced of each ball number, per batter
ball_number_faced_summary <- ball_number_faced |>
  group_by(ball_num_in_over, striker) |>
  summarise(balls_faced = n(), .groups = "drop")

# Dismissals by ball number
dismissals_by_ball_number <- ball_number_faced |>
  select(ball_num_in_over, striker, wicket_type) |>
  filter(wicket_type != "") |>
  group_by(ball_num_in_over, striker) |>
  summarise(dismissals_n = n(), .groups = "drop")

## -----------------------------------------------------------------------------
# Merge data and summarise to league-wide dismissals rate by ball number
dismissals_by_ball_number_summary <- left_join(
  ball_number_faced_summary, dismissals_by_ball_number,
  by = c("ball_num_in_over", "striker")
) |>
  tidyr::replace_na(list(dismissals_n = 0)) |>
  group_by(striker) |>
  mutate(total_balls_faced = sum(balls_faced)) |>
  ungroup() |>
  mutate(dismissals_pct = round(dismissals_n / balls_faced * 100, 2)) |>
  # Include those who have faced more than 200 balls total
  filter(total_balls_faced >= 200) |>
  # Exclude balls beyond 6 - infrequent occurrences
  filter(ball_num_in_over < 7)

# Extract data for specific players
# Healy
dismissals_by_ball_number_summary_healy <- dismissals_by_ball_number_summary |>
  filter(striker == "AJ Healy")
# Mooney
dismissals_by_ball_number_summary_mooney <- dismissals_by_ball_number_summary |>
  filter(striker == "BL Mooney")
# Lanning
dismissals_by_ball_number_summary_lanning <- dismissals_by_ball_number_summary |>
  filter(striker == "MM Lanning")
# Perry
dismissals_by_ball_number_summary_perry <- dismissals_by_ball_number_summary |>
  filter(striker == "EA Perry")
# Devine
dismissals_by_ball_number_summary_devine <- dismissals_by_ball_number_summary |>
  filter(striker == "SFM Devine")
# Knight
dismissals_by_ball_number_summary_knight <- dismissals_by_ball_number_summary |>
  filter(striker == "HC Knight")

## ----out.width="100%"---------------------------------------------------------
# Define consistent plot features ----------------------------------------------
plot_features <- list(
  coord_cartesian(ylim = c(0, 10)),
  theme_minimal(),
  theme(
    text = element_text(family = "roboto_con", colour = "#FFFFFF"),
    plot.title = element_text(
      size = 11, family = "staat", margin = margin(0, 0, 15, 0)
    ),
    plot.subtitle = element_text(
      size = 12, family = "staat", margin = margin(0, 0, 15, 0)
    ),
    plot.caption = element_markdown(
      size = 10, margin = margin(15, 0, 0, 0)
    ),
    axis.text = element_text(size = 9, colour = "#FFFFFF"),
    legend.position = "none",
    panel.grid.major.y = element_line(linetype = "dashed"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(
      fill = "#171F2AFF", colour = NA
    ),
    panel.spacing = unit(2, "lines"),
    plot.margin = unit(c(0.25, 0.25, 0.25, 0.25), "cm")
  ),
  labs(x = NULL, y = NULL)
)

# Build plots ------------------------------------------------------------------
showtext_auto()

# Healy
p1 <- dismissals_by_ball_number_summary |>
  ggplot(aes(x = ball_num_in_over, y = dismissals_pct)) +
  geom_boxplot(
    fill = "#FFFFFF", colour = "#FFFFFF", size = .5,
    alpha = 0.25, notch = TRUE, outlier.shape = NA, coef = 0
  ) +
  geom_point(
    data = dismissals_by_ball_number_summary_healy,
    colour = "#F80F61FF", size = 3
  ) +
  labs(
    title = "WBBL: % dismissals by ball number",
    subtitle = "AJ Healy"
  ) +
  plot_features

# Mooney
p2 <- dismissals_by_ball_number_summary |>
  ggplot(aes(x = ball_num_in_over, y = dismissals_pct)) +
  geom_boxplot(
    fill = "#FFFFFF", colour = "#FFFFFF", size = .5,
    alpha = 0.25, notch = TRUE, outlier.shape = NA, coef = 0
  ) +
  geom_point(
    data = dismissals_by_ball_number_summary_mooney,
    colour = "#FA6900FF", size = 3
  ) +
  labs(subtitle = "BL Mooney") +
  plot_features

# Lanning
p3 <- dismissals_by_ball_number_summary |>
  ggplot(aes(x = ball_num_in_over, y = dismissals_pct)) +
  geom_boxplot(
    fill = "#FFFFFF", colour = "#FFFFFF", size = .5,
    alpha = 0.25, notch = TRUE, outlier.shape = NA, coef = 0
  ) +
  geom_point(
    data = dismissals_by_ball_number_summary_lanning,
    colour = "#018821FF", size = 3
  ) +
  labs(subtitle = "MM Lanning") +
  plot_features

# Perry
p4 <- dismissals_by_ball_number_summary |>
  ggplot(aes(x = ball_num_in_over, y = dismissals_pct)) +
  geom_boxplot(
    fill = "#FFFFFF", colour = "#FFFFFF", size = .5,
    alpha = 0.25, notch = TRUE, outlier.shape = NA, coef = 0
  ) +
  geom_point(
    data = dismissals_by_ball_number_summary_perry,
    colour = "#F80F61FF", size = 3
  ) +
  labs(subtitle = "EA Perry") +
  plot_features

# Devine
p5 <- dismissals_by_ball_number_summary |>
  ggplot(aes(x = ball_num_in_over, y = dismissals_pct)) +
  geom_boxplot(
    fill = "#FFFFFF", colour = "#FFFFFF", size = .5,
    alpha = 0.25, notch = TRUE, outlier.shape = NA, coef = 0
  ) +
  geom_point(
    data = dismissals_by_ball_number_summary_devine,
    colour = "#FA6900FF", size = 3
  ) +
  labs(subtitle = "SFM Devine") +
  plot_features

# Knight
p6 <- dismissals_by_ball_number_summary |>
  ggplot(aes(x = ball_num_in_over, y = dismissals_pct)) +
  geom_boxplot(
    fill = "#FFFFFF", colour = "#FFFFFF", size = .5,
    alpha = 0.25, notch = TRUE, outlier.shape = NA, coef = 0
  ) +
  geom_point(
    data = dismissals_by_ball_number_summary_knight,
    colour = "#95C65CFF", size = 3
  ) +
  labs(
    subtitle = "HC Knight",
    caption = "**Source:** Cricsheet.org // **Plot:** @jacquietran"
  ) +
  plot_features

# Quilt the plots --------------------------------------------------------------
(p1 + p2 + p3) / (p4 + p5 + p6)

## -----------------------------------------------------------------------------
# Subset match metadata for WBB07 games
wbbl07_match_info_tidy <- wbbl_match_info |>
  filter(season == "2021/22", date <= "2021/11/07") |>
  select(
    match_id, winner, winner_runs, winner_wickets, method, outcome,
    eliminator
  ) |>
  mutate(match_id = factor(match_id))

# Subset ball-by-ball data for WBBL07 games
wbbl07_bbb_tidy <- wbbl_bbb |>
  filter(match_id %in% wbbl07_match_info_tidy$match_id) |>
  mutate(
    match_id = factor(match_id),
    runs_scored = runs_off_bat + extras,
    wicket_type = case_when(
      wicket_type == "" ~ NA_character_,
      TRUE ~ wicket_type
    )
  ) |>
  group_by(match_id, innings) |>
  mutate(
    temp_var = 1,
    balls_cumulative = cumsum(temp_var),
    runs_cumulative = cumsum(runs_scored),
    runs_total = max(runs_cumulative)
  ) |>
  ungroup() |>
  select(-temp_var) |>
  # Merge match metadata and ball-by-ball data
  left_join(wbbl07_match_info_tidy, by = "match_id") |>
  mutate(
    outcome_batting_team = case_when(
      outcome %in% c("no result", "tie") ~ as.character(outcome),
      winner == batting_team ~ "won",
      TRUE ~ "lost"
    ),
    outcome_bowling_team = case_when(
      outcome %in% c("no result", "tie") ~ as.character(outcome),
      winner == bowling_team ~ "won",
      TRUE ~ "lost"
    )
  )

## -----------------------------------------------------------------------------
team_strike_rate <- wbbl07_bbb_tidy |>
  # Exclude matches that ended with a Super Over ("tie")
  # and matches that were called off ("no result")
  filter(!outcome_batting_team %in% c("tie", "no result")) |>
  group_by(match_id, innings) |>
  mutate(
    rolling_strike_rate = round(
      runs_cumulative / balls_cumulative * 100, 1
    ),
    wicket_ball_num = case_when(
      !is.na(wicket_type) ~ balls_cumulative,
      TRUE ~ NA_real_
    ),
    wicket_strike_rate = case_when(
      !is.na(wicket_type) ~ rolling_strike_rate,
      TRUE ~ NA_real_
    ),
    innings_description = case_when(
      innings == 1 ~ "Batting 1st",
      innings == 2 ~ "Batting 2nd"
    ),
    bowling_team_short = word(bowling_team, -1),
    start_date_day = lubridate::day(start_date),
    start_date_month = lubridate::month(start_date),
    match_details = glue::glue(
      "{innings_description} vs. {bowling_team_short} ({start_date_day}/{start_date_month})"
    )
  ) |>
  arrange(match_id, innings, balls_cumulative) |>
  mutate(
    match_details = factor(
      match_details,
      levels = unique(match_details)
    ),
    outcome_batting_team = factor(
      outcome_batting_team,
      levels = c("won", "lost")
    )
  )

## ----echo=FALSE, out.width="75%", fig.align="center"--------------------------
knitr::include_graphics("figs/tweet.png")

## ----fig.width=9, fig.height=12, out.width="100%", warning=FALSE, message=FALSE----
# Filter to Renegades' innings only --------------------------------------------
team_strike_rate_renegades <- team_strike_rate |>
  filter(str_detect(batting_team, "Renegades"))

# Build plot -------------------------------------------------------------------
showtext_auto()
team_strike_rate_renegades |>
  ggplot(aes(x = balls_cumulative, y = rolling_strike_rate)) +
  facet_wrap(~match_details, ncol = 3) +
  geom_hline(yintercept = 100, linetype = "dashed", colour = "#CCCCCC") +
  geom_line(aes(colour = outcome_batting_team), linewidth = 1.5) +
  geom_point(
    aes(
      x = team_strike_rate_renegades$wicket_ball_num,
      y = team_strike_rate_renegades$wicket_strike_rate
    ),
    colour = "red", size = 3, alpha = 0.75
  ) +
  labs(
    title = "WBBL07: Melbourne Renegades - Team strike rate (games up to 7 Nov 2021)",
    x = "Ball number in an innings", y = NULL,
    caption = "**Source:** Cricsheet.org // **Plot:** @jacquietran"
  ) +
  scale_x_continuous(breaks = seq(0, 120, by = 30)) +
  scale_color_manual(
    values = c("won" = "#4a8bad", "lost" = "#AD4A8B"),
    labels = c("Renegades won", "Renegades lost")
  ) +
  coord_cartesian(ylim = c(0, 200)) +
  theme_minimal() +
  theme(
    text = element_text(size = 18, family = "roboto_con", colour = "#FFFFFF"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.key.size = unit(1.5, "cm"),
    legend.margin = margin(0, 0, 0, 0),
    legend.spacing.x = unit(0, "cm"),
    legend.spacing.y = unit(0, "cm"),
    plot.title = element_text(family = "staat", margin = margin(0, 0, 15, 0)),
    plot.caption = element_markdown(size = NULL, margin = margin(15, 0, 0, 0)),
    strip.text = element_text(colour = "#FFFFFF", size = 12),
    axis.text = element_text(colour = "#FFFFFF"),
    axis.title.x = element_text(margin = margin(15, 0, 0, 0)),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "#203b60", linetype = "dotted"),
    panel.grid.minor.x = element_blank(),
    plot.background = element_rect(
      fill = "#171F2AFF",
      colour = NA
    ),
    panel.spacing = unit(2, "lines"),
    plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
  )

