import pandas as pd

def read_sector_data(filename):
    sector_data = []
    headers = ['Ticker', 'Date', 'PE Ratio', 'Price to Book Ratio', 'Debt to Equity Ratio', 'Price to FCF Ratio', 'Return on Equity', 'Quarterly_Growth']
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(('Ticker')):
                continue
            else:
                values = line.split()
                sector_data.append(values)
    
    # Convert the sector data into a DataFrame
    if sector_data and headers:
        df = pd.DataFrame(sector_data, columns=headers)
        numeric_columns = headers[2:]  # Exclude 'Ticker' and 'Date'
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df['Ticker'] = df['Ticker'].astype(str)

        return df

    # Return an empty DataFrame if no data is found
    return pd.DataFrame()

# Change this
filename = "utilities_metrics.txt"
test = read_sector_data(filename)

with open(filename, 'w') as file:
    file.write(test.to_string(index=False, header=True, float_format='{:.2f}'.format))
    file.write("\n")