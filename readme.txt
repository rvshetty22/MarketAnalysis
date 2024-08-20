Welcome to Market Analysis! - Rishi Shetty

This program analyzes the top 25 stocks in each sector by market cap, looking at key markey metrics from yfinance and MacroTrends.
Based off a ML alogrithm it estimates the future quarterly percent growth of each stock.
From inputs below the program will recomend the top 3 stocks for each allocated sector.

Inputs for Main.py:
start_date - From when to start collecting data. Most data is only available for the past 10 years.
end_date - Enter the date of last quarter.
collect - Gathers data for input data files. Not neccessary once data is collected. Takes around 10 hours to accumulate all the data.
user_allocation - Manually allocate budget to each sector.
budget - Set investment amount.

Outputs:
Logfile.log - Contains projected growth for each sector and top 3 options even if not enough money is allocated.
plots dir - Shows models accuracy for each sector. R^2 values are very low since data is tough to predict.

Other:
format_fixer.py - After changing filename variable run this program to make output files more readable.
soup.html - Shows html input while scraping. If program crashes and a CloudFlare error is shown here just restart the program.

Need to install a chrome driver for data collection through https://googlechromelabs.github.io/chrome-for-testing/
All scraping is done in accordance to https://www.macrotrends.net/robots.txt