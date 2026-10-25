# IPL auction source files

Copy your existing sold-player CSV files into this folder. Their names must
contain a four-digit auction year, for example:

```text
IPL_Auction_2013_Sold_Player.csv
IPL_Auction_2026_Sold_Player.csv
```

The ETL accepts these source-column alternatives:

| Meaning | Accepted headings |
|---|---|
| Player | `Name`, `Player`, `Player Name` |
| Team | `TeamName`, `Team`, `Team Name`, `Franchise` |
| Role | `Role`, `Player Role` |
| Nationality | `Nationality`, `Country` |
| Winning price | `Winning Bid in Rs`, `Winning Bid in INR`, `Price in rs`, `Price in Rs`, `Price in $`, `Price`, `Winning Bid` |

Do not edit the original source values. Run `auction_etl.py` to produce the
cleaned table under `data/processed/auction/`.
