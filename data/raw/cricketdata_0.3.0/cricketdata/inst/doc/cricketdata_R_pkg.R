## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  echo = TRUE,
  cache = TRUE,
  warning = FALSE
)

## -----------------------------------------------------------------------------
library(cricketdata)
library(dplyr)
library(ggplot2)

## ----getdata, eval=FALSE, echo=FALSE------------------------------------------
# # Avoid downloading the data when the package is checked by CRAN.
# # This only needs to be run once to store the data locally
# ipl_bbb <- fetch_cricsheet("bbb", "male", "ipl")
# wt20 <- fetch_cricinfo("T20", "Women", "Bowling")
# menODI <- fetch_cricinfo("ODI", "Men", "Batting",
#   type = "innings",
#   country = "United States of America"
# )
# meg_lanning_id <- find_player_id("Meg Lanning")$ID
# MegLanning <- fetch_player_data(meg_lanning_id, "ODI") |>
#   mutate(NotOut = (Dismissal == "not out"))
# aus_women <- fetch_player_meta(c(329336, 275487))
# 
# saveRDS(wt20, here::here("inst/extdata/wt20.rds"))
# saveRDS(menODI, here::here("inst/extdata/usmenODI.rds"))
# saveRDS(MegLanning, here::here("inst/extdata/MegLanning.rds"))
# saveRDS(meg_lanning_id, here::here("inst/extdata/meg_lanning_id.rds"))
# saveRDS(ipl_bbb, here::here("inst/extdata/ipl_bbb.rds"))
# saveRDS(aus_women, here::here("inst/extdata/aus_women.rds"))

## ----loaddata, include=FALSE--------------------------------------------------
ipl_bbb <- readRDS("../inst/extdata/ipl_bbb.rds")
wt20 <- readRDS("../inst/extdata/wt20.rds")
menODI <- readRDS("../inst/extdata/usmenODI.rds")
MegLanning <- readRDS("../inst/extdata/MegLanning.rds")
meg_lanning_id <- readRDS("../inst/extdata/meg_lanning_id.rds")
aus_women <- readRDS("../inst/extdata/aus_women.rds")

## ----eval=FALSE---------------------------------------------------------------
# install.packages("cricketdata", dependencies = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("devtools")
# devtools::install_github("robjhyndman/cricketdata")

## ----eval=FALSE, echo=TRUE----------------------------------------------------
# # Fetch all Women's Bowling data for T20 format
# wt20 <- fetch_cricinfo("T20", "Women", "Bowling")

## ----tbl-wt20-----------------------------------------------------------------
# Looking at data
wt20 |>
  glimpse()

# Table showing certain features of the data
wt20 |>
  select(Player, Country, Matches, Runs, Wickets, Economy, StrikeRate) |>
  head() |>
  knitr::kable(
    digits = 2, align = "c",
    caption = "Women Player career profile for international T20"
  )

## ----fig-wt20SRvER, fig.cap="Strike Rate (balls bowled per wicket) Vs Average (runs conceded per wicket) for Women international T20 bowlers. Each observation represents one player, who has taken at least 50 international wickets."----
# Plotting Data
wt20 |>
  filter(Wickets >= 50) |>
  ggplot(aes(y = StrikeRate, x = Average)) +
  geom_point(alpha = 0.3, col = "blue") +
  ggtitle("Women International T20 Bowlers") +
  ylab("Balls bowled per wicket") +
  xlab("Runs conceded per wicket")

## ----echo=TRUE, eval=FALSE----------------------------------------------------
# # Fetch all USA Men's ODI data by innings
# menODI <- fetch_cricinfo("ODI", "Men", "Batting",
#   type = "innings",
#   country = "United States of America"
# )

## ----tbl-USA100s--------------------------------------------------------------
# Table of USA player who have scored a century
menODI |>
  filter(Runs >= 100) |>
  select(Player, Runs, BallsFaced, Fours, Sixes, Opposition) |>
  knitr::kable(digits = 2)

## ----echo=FALSE---------------------------------------------------------------
# menODI |>
#   filter(Runs >= 50) |>
#   ggplot(aes(y = Runs, x = BallsFaced) ) +
#   geom_point(size = 2) +
#   geom_text(aes(label= Player), vjust=-0.5, color="#013369",
#             position = position_dodge(0.9), size=2) +
#   ylab("Runs Scored") + xlab("Balls Faced")

## -----------------------------------------------------------------------------
meg_lanning_id

## ----echo=TRUE, eval=FALSE----------------------------------------------------
# # Fetching the player Meg Lanning's playing data
# MegLanning <- fetch_player_data(meg_lanning_id, "ODI") |>
#   mutate(NotOut = (Dismissal == "not out"))

## ----fig-meglanning, fig.cap="Meg Lanning, Australian captain, has shown amazing consistency over her career, with centuries scored in every year of her career except for 2021, when her highest score from 6 matches was 53."----
dim(MegLanning)
names(MegLanning)

# Compute batting average
MLave <- MegLanning |>
  filter(!is.na(Runs)) |>
  summarise(Average = sum(Runs) / (n() - sum(NotOut))) |>
  pull(Average)
names(MLave) <- paste("Average =", round(MLave, 2))

# Plot ODI scores
ggplot(MegLanning) +
  geom_hline(aes(yintercept = MLave), col = "gray") +
  geom_point(aes(x = Date, y = Runs, col = NotOut)) +
  ggtitle("Meg Lanning ODI Scores") +
  scale_y_continuous(sec.axis = sec_axis(~., breaks = MLave))

## ----echo=TRUE, eval=FALSE----------------------------------------------------
# # Fetch all IPL ball-by-ball data
# ipl_bbb <- fetch_cricsheet("bbb", "male", "ipl")

## -----------------------------------------------------------------------------
ipl_bbb |>
  glimpse()

## ----fig-iplbatter, fig.cap="Top 20 prolific batters in IPL 2022. We show what percentage of balls they hit for a boundary (4 or 6) against percentage of how many balls they do not score off of (dot percent). Ideally we want to be in top left quadrant, high boundary % and low dot %."----
# Top 20 batters wrt Boundary and Dot % in IPL 2022 season
ipl_bbb |>
  filter(season == "2022") |>
  group_by(striker) |>
  summarize(
    Runs = sum(runs_off_bat), BallsFaced = n() - sum(!is.na(wides)),
    StrikeRate = Runs / BallsFaced, DotPercent = sum(runs_off_bat == 0) * 100 / BallsFaced,
    BoundaryPercent = sum(runs_off_bat %in% c(4, 6)) * 100 / BallsFaced
  ) |>
  arrange(desc(Runs)) |>
  rename(Batter = striker) |>
  slice(1:20) |>
  ggplot(aes(y = BoundaryPercent, x = DotPercent, size = BallsFaced)) +
  geom_point(color = "red", alpha = 0.3) +
  geom_text(aes(label = Batter),
    vjust = -0.5, hjust = 0.5, color = "#013369",
    position = position_dodge(0.9), size = 3
  ) +
  ylab("Boundary Percent") +
  xlab("Dot Percent") +
  ggtitle("IPL 2022: Top 20 Batters")

## ----tbl-IPL2022Batters-------------------------------------------------------
# Top 10 prolific batters in IPL 2022 season.
ipl_bbb |>
  filter(season == "2022") |>
  group_by(striker) |>
  summarize(
    Runs = sum(runs_off_bat), BallsFaced = n() - sum(!is.na(wides)),
    StrikeRate = Runs / BallsFaced,
    DotPercent = sum(runs_off_bat == 0) * 100 / BallsFaced,
    BoundaryPercent = sum(runs_off_bat %in% c(4, 6)) * 100 / BallsFaced
  ) |>
  arrange(desc(Runs)) |>
  rename(Batter = striker) |>
  slice(1:10) |>
  knitr::kable(digits = 1, align = "c")

## ----tbl-playermetadata-------------------------------------------------------
player_meta |>
  filter(!is.na(playing_role)) |>
  select(-cricinfo_id, -unique_name, -name) |>
  head() |>
  knitr::kable(
    digits = 1, align = "c", format = "pipe",
    col.names = c(
      "ID", "FullName", "Country", "DOB", "BirthPlace",
      "BattingStyle", "BowlingStyle", "PlayingRole"
    )
  )

## ----echo=TRUE, eval=FALSE----------------------------------------------------
# # Download meta data on Meg Lanning and Ellyse Perry
# aus_women <- fetch_player_meta(c(329336, 275487))

## ----tbl-ausplayermetadata----------------------------------------------------
aus_women |>
  select(-name) |>
  knitr::kable(
    digits = 1, align = "c", format = "pipe",
    col.names = c(
      "ID", "FullName", "Country", "DOB", "BattingStyle", "BowlingStyle"
    )
  )

