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

A). Hourly Sales
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
``` python
# Plot of peak sales hours
plt.figure(figsize=(18, 9))
sns.scatterplot(x="hour", y="Total_Price", data=hourly_sales, palette="Blues")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Sales")
plt.title("Peak Sales Hours")
plt.xticks(range(0, 24))
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()
```
![image](https://github.com/user-attachments/assets/cd443250-a37b-4913-86ed-1aaf80bbb9a2)

B). Weekly Sales
``` python
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors='coerce')
df["week_number"] = df["transaction_date"].dt.isocalendar().week

weekly_sales = df.groupby("week_number")["Total_Price"].sum().reset_index()

# Plot weekly sales
plt.figure(figsize=(14, 8))
sns.lineplot(data=weekly_sales, x="week_number", y="Total_Price", marker="o", linewidth=2, color="green")
plt.xlabel("Week Number")
plt.ylabel("Total Sales ($)")
plt.title("Total Sales per Week")
plt.xticks(range(weekly_sales["week_number"].min(), weekly_sales["week_number"].max() + 1))
plt.grid(False)
plt.show()
```
![image](https://github.com/user-attachments/assets/b5718029-3806-4e1e-9549-1843f33e27ea)
weekly_sales.sort_values(by ='Total_Price' ,ascending = False)
![image](https://github.com/user-attachments/assets/439c1b1f-a38d-43fa-8ee7-bb45fdb9c823)
``` python
print(weekly_sales.max())
print('\t')
print(weekly_sales.min())
```
![image](https://github.com/user-attachments/assets/2abf95e1-52f7-481e-a1c6-df642536f047)
``` python
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors='coerce')

df["week_number"] = df["transaction_date"].dt.isocalendar().week
df["weekday"] = df["transaction_date"].dt.day_name()

weekly_sales_heatmap = df.pivot_table(values="Total_Price", index="week_number", columns="weekday", aggfunc="sum")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly_sales_heatmap = weekly_sales_heatmap[weekday_order]

plt.figure(figsize=(12, 8))
sns.heatmap(weekly_sales_heatmap, cmap="YlGnBu", linewidths= 0.5, annot=True, fmt=".0f")
plt.xlabel("Day of the Week")
plt.ylabel("Week Number")
plt.title("Weekly Sales Heatmap")
plt.show()
```
![image](https://github.com/user-attachments/assets/607733e2-724c-4ce6-b0db-5a91eeb2c40d)
``` python
weekly_sales_heatmap.sum()
```
![image](https://github.com/user-attachments/assets/9f9d96f3-5daf-4a1f-93f6-2e964145656b)

C). Product wise Sales
``` python
Product_sales = df.groupby('product_category')['Total_Price'].sum().sort_values(ascending = False)
Product_sales.reset_index()
```
![image](https://github.com/user-attachments/assets/cdc56da8-5441-450d-8df2-2b257c9e5596)

D). Seasonal wise Sales Perfromance
``` python
df['month'] = df['transaction_date'].dt.month

monthly_sales = df.groupby('month')['Total_Price'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='Total_Price', data=monthly_sales, marker='o')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Trend')
plt.xticks(range(1, 13))
plt.show()
```
![image](https://github.com/user-attachments/assets/04cb6c97-b0fb-42d8-8b4e-071895ae201a)
``` python
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

monthly_sales['month_name'] = monthly_sales['month'].map(month_names)
monthly_sales = monthly_sales.rename(columns={'month': 'month_number'})
pd.pivot_table(monthly_sales, index = 'month_name', values = 'Total_Price').reset_index()
```
![image](https://github.com/user-attachments/assets/2063bc0e-a9ca-4943-b0b9-2ac231d23009)
``` python
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5, 6]:
        return 'Summer'
    elif month in [7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Autumn'
    else:
        return 'Unknown'

monthly_sales['Season'] = monthly_sales['month_name'].apply(lambda x: assign_season(list(month_names.keys())[list(month_names.values()).index(x)]))
monthly_sales
```
![image](https://github.com/user-attachments/assets/0a341400-99d1-414f-a88a-aa635a98d109)
``` python
plt.figure(figsize=(12, 6))
sns.barplot(x='Season', y='Total_Price', data=monthly_sales, palette="viridis")
plt.xlabel('Seasons')
plt.ylabel('Total Sales')
plt.title('Seasonal Sales Trend')
plt.tight_layout()

# Text Values
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{height:.0f}',
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', 
                       fontsize=12, color='black',
                       xytext=(0, 5), textcoords='offset points')
plt.show()
```
![image](https://github.com/user-attachments/assets/183d3ffb-5299-4147-83de-a31d16d65afd)
``` python
seasons_table = pd.pivot_table(monthly_sales, index = 'Season', values = 'Total_Price').reset_index()
seasons_table.index = list(range(1,seasons_table.shape[0]+1))
seasons_table
```
![image](https://github.com/user-attachments/assets/82bde03a-1ede-467e-9fa3-32cf96847937)

### Pricing Strategy & Revenue Optimization






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
