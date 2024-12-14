import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://root:Dhruv001@localhost/csv_sql"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Orders Per Region
query = 'SELECT Region, COUNT(DISTINCT OrderID) AS Quantity FROM Data GROUP BY Region ORDER BY Quantity'
df = pd.read_sql(query, engine)
fig = px.pie(df, values = 'Quantity', names = 'Region', title = 'Orders Per Region')
fig.show()

# Sales Over Time
query = 'SELECT OrderDate, Sales FROM Data ORDER BY OrderDate'
df = pd.read_sql(query, engine)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
fig = px.line(df, x = 'OrderDate', y = 'Sales', title = 'Sales Over Time')
fig.show()

# Profit Per Subcategory
query = 'SELECT Subcategory, SUM(Profit) AS Profit FROM Data GROUP BY Subcategory Order BY Profit DESC'
df = pd.read_sql(query, engine)
fig = px.bar(df, x = 'Subcategory', y = 'Profit', title = 'Total Profit Per Subcategory')
fig.show()

# Sales Per Customer Segment
query = 'SELECT Segment, SUM(Sales) AS Sales FROM Data GROUP BY Segment ORDER BY Sales DESC'
df = pd.read_sql(query, engine)
fig = px.bar(df, x = 'Segment', y = 'Sales', title = 'Total Sales Per Segment')
fig.show()

# Sales Per Product
query = 'SELECT ProductName, SUM(Sales) AS Sales FROM Data Group BY ProductName ORDER BY Sales DESC LIMIT 25'
df = pd.read_sql(query, engine)
fig = px.bar(df, x = 'ProductName', y = 'Sales', title = 'Total Sales Per Product (Top 25)')
fig.show()

# Frequency of Shipping Mode
query = 'SELECT ShipMode, COUNT(ShipMode) AS Frequency FROM Data GROUP BY ShipMode ORDER BY Frequency DESC'
df = pd.read_sql(query, engine)
fig = px.bar(df, x = 'ShipMode', y = 'Frequency', title = 'Frequency of Shipping Modes')
fig.show()

# Total Profit per Customer
query = 'SELECT CustomerName, SUM(Profit) AS Profit FROM Data GROUP BY CustomerName ORDER BY Profit DESC LIMIT 25'
df = pd.read_sql(query, engine)
fig = px.bar(df, x = 'CustomerName', y = 'Profit', title = 'Total Profit Per Customer (Top 25)')
fig.show()

# Customers Per City
query = 'SELECT COUNT(DISTINCT CustomerName) AS Customers, City FROM Data GROUP BY City ORDER BY Customers DESC LIMIT 25'
df = pd.read_sql(query, engine)
fig = px.bar(df, x = 'City', y = 'Customers', title = 'Total Customers Per City (Top 25)')
fig.show()

session.close()