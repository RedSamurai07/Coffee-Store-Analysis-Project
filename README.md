# Coffee Store Analysis

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Data Analysis](#data-analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
- The coffee shop industry is highly competitive, and optimizing sales, pricing, and customer behavior insights is crucial for business success. This project aims to analyze historical sales data to identify key trends, customer preferences, and factors influencing sales performance. The goal is to provide data-driven recommendations to enhance revenue, optimize inventory, and improve customer experience.

### Executive Summary
Key names with insights

### 🎯Goal
The objective of this analysis is to:
1. Understand Sales Patterns – Identify peak sales hours, best-selling products, and seasonal trends.
2. Optimize Pricing Strategy – Analyze the impact of pricing on sales volume.
3. Improve Inventory Management – Reduce wastage by forecasting demand based on historical sales data.
4. Enhance Customer Experience – Segment customers based on purchasing behavior to improve marketing strategies.
5. Predict Future Sales – Use statistical and machine learning techniques to forecast demand.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1GK4tnY4_YfX8ccNhVtedEoGpsUUGOwzlysEVEQr_xpA/edit?gid=1748548740#gid=1748548740)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| transaction_id | Unique identifier for each transaction. | int |
| date | Date when the transaction occurred (MM/DD/YYYY format). | object |
| transaction_time | Time when the transaction was made (HH:MM:SS format).| object |
| transaction_qty  | Number of units purchased in the transaction. | int |
| store_id  | Unique identifier for the store where the purchase was made. | int |       
| store_location |  Name of the store location. | object |
| product_id     |  Unique identifier for each product sold. | int |
| unit_price     |  Price per unit of the product (includes currency symbol).| float |
| product_category | Broad classification of the product. | object |
| product_type     | Subcategory of the product | object |
| product_detail   | Detailed name of the specific product. | object |

### Tools
 - Excel
 - SQL
 - Python
  
### Data Analysis
- Importing Libraries
``` python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import warnings
  warnings.filterwarnings('ignore')
```
- Loading the dataset
``` python  
df = pd.read_csv('transactions.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/c5e51c21-09a6-4dd7-b628-69771407ba7f)
- Dimension and Shape of the dataset
``` python
df.ndim
```
![image](https://github.com/user-attachments/assets/c5059414-7df6-4586-85e3-57a5a792a862)
``` python
df.shape
```
![image](https://github.com/user-attachments/assets/173a1f23-16ac-43f4-9898-542ff7547ac2)
- Information of the Dataset
``` python
df.info()
```
![image](https://github.com/user-attachments/assets/c6fbebc9-0ce5-498b-9430-acf27c86cce3)
- Dropping unwanted columns for analysis
``` python
df.drop('product_detail',axis = 1, inplace = True)
df.index = list(range(1,df.shape[0]+1))
```
- Checking for Null/Nan values in all the columns and rows
``` python
df.isna().sum()/len(df)*100
```
![image](https://github.com/user-attachments/assets/6bee514d-7554-47ae-a513-4e9bf310fa4f)
``` python
df.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/1a9b4b8a-28b7-4e0e-ae98-b80e483c9da2)
- Feature Engineering
``` python
df['Total_Price'] = df['transaction_qty'] * df['unit_price']
customer_spending = df.groupby("transaction_id")["Total_Price"].sum().reset_index()
customer_spending["spending_segment"] = pd.qcut(customer_spending["Total_Price"], q=3, labels=["Low", "Medium", "High"])
df = df.merge(customer_spending[["transaction_id", "spending_segment"]], on="transaction_id", how="left")
df.head()
```
![image](https://github.com/user-attachments/assets/4da663f6-19b6-4bf0-8f59-ca440cb69464)
```python
pd.crosstab(df['transaction_id'], df['spending_segment']).sum()
```
![image](https://github.com/user-attachments/assets/10db472f-affb-4eda-80d4-3db26ee1c10d)
- Converting index from 1 to N
``` python
df.index = list(range(1,df.shape[0]+1))
df
```
![image](https://github.com/user-attachments/assets/207bdc5e-e79a-48c4-8680-a05ecf37b7c3)
- Descriptive Statistics
``` python
df.describe()
```
![image](https://github.com/user-attachments/assets/bc2fcdb7-3bf9-46ea-9af3-b108d3b6eeb5)
``` python
df.select_dtypes(include = 'object').describe()
```
![image](https://github.com/user-attachments/assets/530c3a20-a71e-4ab6-bffb-27cb75486901)\
**Descriptive Statistics**

1.   The maximum transactions was on 27th March 2025 at 8:19 AM for about 22
     times mostly occured.
2.   Hell's kitchen had the highest profitable location.
3.   The most popular category were Coffee was the most sold for about 21589.
4.   Brewed Chai tea was the most frequently bought by the consumers at the coffee store.

### Sales performance

``` python
df["transaction_date"] = pd.to_datetime(df["date"], errors='coerce')
df["weekday"] = df["transaction_date"].dt.day_name()
weekday_sales = df.groupby("weekday")["Total_Price"].sum().reset_index()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_sales = weekday_sales.set_index("weekday").reindex(weekday_order).reset_index()
```
``` python
df['hour'] = df['transaction_time']
hourly_sales = df.groupby('hour')['Total_Price'].sum().reset_index()
hourly_sales.index = list(range(1,hourly_sales.shape[0]+1))
hourly_sales
```
![image](https://github.com/user-attachments/assets/3f994075-ae13-4fe2-b3b6-334784e30e8e)
``` python
hourly_sales = df.groupby("hour")["Total_Price"].sum().reset_index()
```



### Insights
- The highests weekly sales was on week number 31.
- In weekdays Monday's, Wednesday & Friday had the most sales and on weekends Sunday had the moore number of the sales.
- Coffee & Tea was the most frequently product category sold.
- The highest montly sales were on Aug, Sep, Oct, Nov, Dec
- The Highest quantity were sold in Monson, Autumn & Winter Season.
- The products with most quantity sold were Barista Expresso, Brewed Chai tea, Hot chocolate,Gourmet brewed coffee, brewed black tea.
- The store location with the highest revenue is Hell's Kitchen.
- Products with the highest stock wastage due to low demand are Branded and Packaged Chocolate

### Recommendations
- Recommneded Inventory for the Coffee Store:*
- Coffee, Tea, Bakery, Drinking Chocolate, Flavours, Coffee beans, Loose Tea.
- Reduce the purcahse of Branded, packaged chocolate and sell it only on occassions.
-  Discount do affect the product Sales with respect to 0%, 10%, 20%. Ideally 10% recommended to attract more customers.*
-  On an average, we need to provide discounts or combo offers on these recommended products either on one of the weekdays or weekends.*
-  Introduce different flavours on different locations for more visit.
- Coffee, Tea & Drinking Chocolate should be the most highlight on the menu with offers on specialb  occasions.*
- Barista Espresso, Brewed Chai tea and Gourmet brewed coffee to be recommended more to professionals and elderly citizens to reduce type 2 diabetes and some cancers,as well as to improvr cognitive memory.
