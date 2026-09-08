## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  echo = TRUE
)
# Okabi-Ito colours
options(
  ggplot2.discrete.colour = c("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442")
)

## ----setup--------------------------------------------------------------------
library(cricketdata)
library(dplyr)
library(ggplot2)

## ----getdata, eval=FALSE, echo=FALSE------------------------------------------
# # Avoid downloading the data when the package is checked by CRAN.
# # This only needs to be run once to store the data locally
# wt20 <- fetch_cricinfo("T20", "Women", "Bowling")
# menODI <- fetch_cricinfo("ODI", "Men", "Batting", type = "innings", country = "Australia")
# Indfielding <- fetch_cricinfo("Test", "Men", "Fielding", country = "India")
# meg_lanning_id <- find_player_id("Meg Lanning")$ID
# MegLanning <- fetch_player_data(meg_lanning_id, "ODI") |>
#   mutate(NotOut = (Dismissal == "not out")) |>
#   mutate(NotOut = tidyr::replace_na(NotOut, FALSE))
# 
# saveRDS(wt20, here::here("inst/extdata/wt20.rds"))
# saveRDS(menODI, here::here("inst/extdata/menODI.rds"))
# saveRDS(Indfielding, here::here("inst/extdata/Indfielding.rds"))
# saveRDS(MegLanning, here::here("inst/extdata/MegLanning.rds"))

## ----loaddata, include=FALSE--------------------------------------------------
wt20 <- readRDS("../inst/extdata/wt20.rds")
menODI <- readRDS("../inst/extdata/menODI.rds")
Indfielding <- readRDS("../inst/extdata/Indfielding.rds")
MegLanning <- readRDS("../inst/extdata/MegLanning.rds")

## ----woment20, message=FALSE, echo = FALSE------------------------------------
wt20 |>
  head() |>
  knitr::kable(digits = 2)

## ----woment20graph, fig.width=10, fig.height=8--------------------------------
wt20 |>
  filter(Wickets > 20, !is.na(Country)) |>
  ggplot(aes(y = StrikeRate, x = Country)) +
  geom_boxplot() +
  geom_point(alpha = 0.3, col = "blue") +
  ggtitle("Women T20: Strike Rates") +
  ylab("Balls per wicket") +
  coord_flip()

## ----menodi, message=FALSE, echo=FALSE----------------------------------------
menODI |>
  head() |>
  knitr::kable()

## ----menodigraph, warning=FALSE, message=FALSE--------------------------------
menODI |>
  ggplot(aes(y = Runs, x = Date)) +
  geom_point(alpha = 0.2, col = "#D55E00") +
  geom_smooth() +
  ggtitle("Australia Men ODI: Runs per Innings")

## ----indiafielding, echo=FALSE------------------------------------------------
Indfielding |>
  head() |>
  knitr::kable()

## ----indiafieldinggraph-------------------------------------------------------
Indfielding |>
  mutate(wktkeeper = (CaughtBehind > 0) | (Stumped > 0)) |>
  ggplot(aes(x = Matches, y = Dismissals, col = wktkeeper)) +
  geom_point() +
  ggtitle("Indian Men Test Fielding")

## ----meglanning, echo=FALSE---------------------------------------------------
MegLanning |>
  head() |>
  knitr::kable()

## ----meglanninggraph----------------------------------------------------------
# Compute batting average
MLave <- MegLanning |>
  summarise(
    Innings = sum(!is.na(Runs)),
    Average = sum(Runs, na.rm = TRUE) / (Innings - sum(NotOut, na.rm=TRUE))
  ) |>
  pull(Average)
names(MLave) <- paste("Average =", round(MLave, 2))
# Plot ODI scores
ggplot(MegLanning) +
  geom_hline(aes(yintercept = MLave), col = "gray") +
  geom_point(aes(x = Date, y = Runs, col = NotOut)) +
  ggtitle("Meg Lanning ODI Scores") +
  scale_y_continuous(sec.axis = sec_axis(~., breaks = MLave))

