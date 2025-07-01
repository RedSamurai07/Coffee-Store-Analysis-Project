![image](https://github.com/user-attachments/assets/0297ae75-a0ce-47d3-84c1-16c4c41d0c03)![image](https://github.com/user-attachments/assets/fde6a4f0-feec-4da1-80b6-019b8c3ae92e)# Coffee Store Analysis

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
1). Python

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
``` pythonseasonal_sales = monthly_sales.groupby('Season')['Total_Price'].sum()

plt.figure(figsize=(8, 8))
plt.pie(seasonal_sales, labels=seasonal_sales.index, 
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(seasonal_sales)))
plt.title('Seasonal Sales Distribution')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/2cbed06d-2de2-4405-98d0-9451a47803d7)
``` python
seasons_table = pd.pivot_table(monthly_sales, index = 'Season', values = 'Total_Price').reset_index()
seasons_table.index = list(range(1,seasons_table.shape[0]+1))
seasons_table
```
![image](https://github.com/user-attachments/assets/82bde03a-1ede-467e-9fa3-32cf96847937)

### Pricing Strategy & Revenue Optimization
``` python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='unit_price', y='transaction_qty', data=df)
plt.xlabel('Unit Price')
plt.ylabel('Transaction Quantity')
plt.title('Relationship between Unit Price and Transaction Quantity')
plt.show()

correlation = df['unit_price'].corr(df['transaction_qty'])
print(f"Correlation between Unit Price and Transaction Quantity: {correlation}")
```
![image](https://github.com/user-attachments/assets/c620bf3f-5d40-4808-8713-4bb87620fd0e)
- There is no co-orelation between Transaction Quamntity and Unit price.
``` python
transact_table = pd.pivot_table(df, index = 'transaction_qty', values = 'unit_price').reset_index()
transact_table.index = list(range(1,transact_table.shape[0]+1))
transact_table
```
![image](https://github.com/user-attachments/assets/774f4c5f-109b-4719-82ab-2f72a3af6600)
``` python
aggregated_sales = df.groupby(['product_type'])['Total_Price'].sum().reset_index().sort_values(by = 'Total_Price',ascending = False)

plt.figure(figsize=(18, 8))
sns.barplot(x='product_type', y='Total_Price', data=aggregated_sales)
plt.title('Sales by product Category')
plt.xlabel('Products')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center_baseline',
                       fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()
```
![image](https://github.com/user-attachments/assets/e1fd404f-4bb2-4724-9a10-02582ed44d65)
``` python
product_performance = df.groupby('product_category').agg({'Total_Price': 'sum', 'unit_price': 'mean'}).reset_index()
product_performance['sales_per_unit_price'] = product_performance['Total_Price'] / product_performance['unit_price']
product_performance = product_performance.sort_values('unit_price', ascending=False)
product_performance.index = list(range(1,product_performance.shape[0]+1))
print(product_performance)

plt.figure(figsize=(10, 5))
sns.scatterplot(x='unit_price', y='Total_Price', data=product_performance, size='sales_per_unit_price', sizes=(20, 200))
plt.xlabel('Average Unit Price')
plt.ylabel('Total Sales')
plt.title('Product Performance by Price and Sales')
plt.show()
```
![image](https://github.com/user-attachments/assets/8a1c8316-fb27-4b23-ab99-dccc0b683b13)

### Customer Behavior Analysis
A). Market Basket Analysis
``` python
customer_frequency = df.groupby("transaction_id")["transaction_date"].count().reset_index()
customer_frequency.rename(columns={"transaction_date": "purchase_frequency"}, inplace=True)

customer_frequency["frequency_segment"] = pd.cut(
    customer_frequency["purchase_frequency"],
    bins=3,
    labels=["Occasional", "Regular", "Frequent"]
)
customer_spending = df.groupby("transaction_id")["Total_Price"].sum().reset_index()
customer_spending.rename(columns={"Total_Price": "total_spent"}, inplace=True)

customer_segments = pd.merge(customer_frequency, customer_spending, on="transaction_id", how="left")

customer_segments["spending_segment"] = pd.cut(
    customer_segments["total_spent"],
    bins=3,
    labels=["Low Spender", "Medium Spender", "High Spender"]
)
crosstab_segments = pd.crosstab(
    customer_segments["spending_segment"],
    customer_segments["frequency_segment"]
)
```
``` python
basket = df.groupby('product_type')[['transaction_qty','Total_Price']].sum().reset_index().sort_values(by ='transaction_qty',ascending = False)
basket.index = list(range(1,basket.shape[0]+1))
basket.head(5)
```
![image](https://github.com/user-attachments/assets/d314e52a-c611-4d1e-b8f7-2977199ebbe4)
``` python
plt.figure(figsize=(12, 6))
sns.barplot(hue='product_type', y='transaction_qty', data=basket.head())
plt.xlabel('Product Type')
plt.ylabel('Transaction Quantity')
plt.title('Product Types by Transaction Quantity')
plt.xticks(rotation=45, ha='right')
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center_baseline',
                       fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()
```
![image](https://github.com/user-attachments/assets/bcf33959-9a8b-42f6-9e20-a025574a9b35)
``` python
hourly_prod_sales = df.groupby('hour')['Total_Price'].sum().reset_index().sort_values(by = 'Total_Price', ascending = False)
sns.lineplot(data=hourly_prod_sales, x="hour", y="Total_Price", marker="o", linewidth=2, color="green")
hourly_prod_sales
```
![image](https://github.com/user-attachments/assets/0221c53c-9bc4-41b7-93a9-21488229790c)
one more graph need to be added
``` python
product_transaction_counts = df.groupby('product_type')['transaction_id'].count().reset_index().sort_values(by = 'transaction_id',ascending = False)
product_transaction_counts.index = list(range(1,product_transaction_counts.shape[0]+1))
product_transaction_counts.head()
```
![image](https://github.com/user-attachments/assets/731534ae-045a-4e7f-8d16-630619c27ae3)
``` python
plt.figure(figsize=(12, 6))
sns.barplot(hue='product_type', y='transaction_id', data=product_transaction_counts.sort_values(by = 'transaction_id',ascending = False).head())
plt.xlabel('Product Type')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions per Product Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center_baseline',
                       fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()
```
![image](https://github.com/user-attachments/assets/ecf714d3-fedb-45c7-af1d-32333802e559)

### Store Location Performance Analysis
``` python
store_revenue = df.groupby('store_location')['Total_Price'].sum()

# store with the highest revenue
highest_revenue_store = store_revenue.idxmax()
highest_revenue = store_revenue.max()

# store with the lowest revenue
lowest_revenue_store = store_revenue.idxmin()
lowest_revenue = store_revenue.min()

print(f"Store with highest revenue: {highest_revenue_store} (${highest_revenue:.2f})")
print(f"Store with lowest revenue: {lowest_revenue_store} (${lowest_revenue:.2f})")
```
![image](https://github.com/user-attachments/assets/22024e14-97a3-42b3-a2da-03358e3577de)
``` python
store_performance = df.groupby('store_location').agg({'Total_Price': 'sum', 'transaction_id': 'count'})
store_performance['Average_Transaction_Value'] = store_performance['Total_Price'] / store_performance['transaction_id']
print(store_performance)

plt.figure(figsize=(10, 6))
sns.barplot(x=store_performance.index, y='Average_Transaction_Value', data=store_performance)
plt.xlabel('Store Location')
plt.ylabel('Average Transaction Value')
plt.title('Average Transaction Value per Store Location')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/22a258cf-6bd1-4be7-8a44-74156fa41e79)

### Inventory & Waste Management
``` python
product_sales = df.groupby('product_category')['transaction_qty'].sum()
low_demand_products = product_sales[product_sales < product_sales.quantile(0.25)].reset_index()
print("Products with the highest stock wastage due to low demand:")
low_demand_products.index = list(range(1,low_demand_products.shape[0]+1))
low_demand_products
```
![image](https://github.com/user-attachments/assets/60c87df0-5960-4cba-82e2-4ba9d2ad657c)
``` python
average_weekly_sales = df.groupby(['product_category', 'week_number'])['transaction_qty'].sum().groupby('product_category').mean()
# Setting a safety stock level (You might need to adjust this based on our business needs and product characteristics).
safety_stock = 2

# Calculating the recommended inventory level.
recommended_inventory = (average_weekly_sales * safety_stock).astype(int).reset_index().sort_values(by ='transaction_qty', ascending = False )

print("Recommended Inventory Levels:")
recommended_inventory.index = list(range(1,recommended_inventory.shape[0]+1))
recommended_inventory
```
![image](https://github.com/user-attachments/assets/2e6e6148-e8c5-4d18-8281-f057f0e9d50e)
``` python
plt.figure(figsize=(12, 6))
sns.barplot(hue='product_category', y='transaction_qty', data=recommended_inventory.sort_values(by = 'transaction_qty',ascending = False).head())
plt.xlabel('Product Type')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions per Product Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center_baseline',
                       fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()
```
![image](https://github.com/user-attachments/assets/d9081cce-f5a7-4d4c-b555-6ca7631862c9)

### Marketing & Promotion Effectiveness
```
np.random.seed(42)
df['discount'] = np.random.choice([0, 0.1, 0.15, 0.2], size=len(df)) # Simulating 0%, 10%, or 20% discount

df['discounted_price'] = df['unit_price'] * (1 - df['discount'])
df['discounted_total_price'] = df['transaction_qty'] * df['discounted_price']

discount_sales = df.groupby('discount')['discounted_total_price'].sum().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(x='discount', y='discounted_total_price', data=discount_sales)
plt.xlabel('Discount Percentage')
plt.ylabel('Total Discounted Sales')
plt.title('Impact of Discounts on Total Sales')
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center_baseline',
                       fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()
print(discount_sales)
```
![image](https://github.com/user-attachments/assets/c4f887d3-63e5-4184-bb1c-4ad0a3bad367)
Discount vs Without Discount
``` python
# Creating a new column to categorize transactions
df['has_discount'] = df['discount'].apply(lambda x: 'With Discount' if x > 0 else 'Without Discount')

# Calculating average transaction value for each category
avg_transact_value = df.groupby('has_discount')['discounted_total_price'].mean().reset_index()

plt.figure(figsize=(10,7))
sns.barplot(x='has_discount', y='discounted_total_price', data=avg_transact_value, palette='coolwarm')

plt.xlabel('Discount Category')
plt.ylabel('Average Transaction Value')
plt.title('Average Transaction Value: With vs Without Discount')
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{height:.0f}',
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom',
                       fontsize=12, color='black',
                       xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()
print(avg_transact_value)
```
![image](https://github.com/user-attachments/assets/5434441c-5e77-46cf-b694-028ccc601b14)
``` python
highest_sales_discount = discount_sales.loc[discount_sales['discounted_total_price'].idxmax()]
print(f"Discount with highest total sales: {highest_sales_discount}")
```
![image](https://github.com/user-attachments/assets/1b302a02-27b1-40fe-95ce-9f1788eb2621)
``` python
highest_avg_transaction_discount = avg_transact_value.loc[avg_transact_value['discounted_total_price'].idxmax()]
print(f"Discount with highest average transaction value: {highest_avg_transaction_discount}")
```
![image](https://github.com/user-attachments/assets/09e689f4-2dba-4b41-9add-3df61b2b071c)

``` python
# To Analyze average transaction value with and without discounts
avg_transact_value = df.groupby('discount')['discounted_total_price'].mean().reset_index()

# To Find out the discount level with the highest average transaction value
highest_avg_transaction_discount = avg_transact_value.loc[avg_transact_value['discounted_total_price'].idxmax()]
print(f"Discount with highest average transaction value: {highest_avg_transaction_discount}")
```
![image](https://github.com/user-attachments/assets/4e51db79-ecd3-4700-a370-2de97480d935)

## Hypothesis testing:

1). Does the average sales revenue differ between weekends and weekdays?
``` python
#H₀ (Null Hypothesis): There is no significant difference in average sales revenue between weekends and weekdays.
#Ha (Alternative Hypothesis): There is a significant difference in average sales revenue between weekends and weekdays.

import scipy.stats as stats
weekend_mask = df['weekday'].isin(['Saturday', 'Sunday'])

# Separation of sales data for weekends and weekdays
weekend_sales = df.loc[weekend_mask, 'Total_Price']
weekday_sales = df.loc[~weekend_mask, 'Total_Price']

# We Perform an independent samples t-test
t_statistic, p_value = stats.ttest_ind(weekend_sales, weekday_sales)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in average sales revenue between weekends and weekdays.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in average sales revenue between weekends and weekdays.")
```
![image](https://github.com/user-attachments/assets/71cadaf6-7b81-4aba-90ac-0e13d88db84d)

2). Does the average sales revenue differ across product categories?
``` python
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Total_Price ~ C(product_category)', data=df).fit()
# Anova 
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

alpha = 0.05
if anova_table['PR(>F)'][0] < alpha:
    print("Reject the null hypothesis. There is a significant difference in average sales revenue across product categories.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in average sales revenue across product categories.")
```
![image](https://github.com/user-attachments/assets/dbc27620-9277-4796-ab45-849c3bddcb4b)

3). Does the average quantity sold change significantly based on price range?
``` python
#H₀: There is no significant difference in quantity sold across different price ranges.
#Ha: There is a significant difference in quantity sold across different price ranges.

#Calculating the correlation between unit price and transaction quantity
correlation = df['unit_price'].corr(df['transaction_qty'])
print(f"Correlation between Unit Price and Transaction Quantity: {correlation}")

# Creation of price bins
price_bins = [0, 10, 20, 30, 40, float('inf')]  # Define appropriate price ranges
price_labels = ['0-10', '10-20', '20-30', '30-40', '40+']
df['price_range'] = pd.cut(df['unit_price'], bins=price_bins, labels=price_labels, right=False)

# Calculating average quantity sold per price range
avg_qty_by_price = df.groupby('price_range')['transaction_qty'].mean()
print(avg_qty_by_price)

# Performing Anova test
model = ols('transaction_qty ~ C(price_range)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

alpha = 0.05
if anova_table['PR(>F)'][0] < alpha:
  print("Reject the null hypothesis. There is a statistically significant difference in average quantity sold across price ranges.")
else:
  print("Fail to reject the null hypothesis. There is no statistically significant difference in average quantity sold across price ranges.")
```
![image](https://github.com/user-attachments/assets/0a917fe3-1f4d-4361-a414-4702d9a3a4cc)

4). Do different store locations have significantly different sales revenue?
``` python
#H₀: There is no significant difference in sales revenue across different stores.
#Ha: There is a significant difference in sales revenue across different stores.

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('Total_Price ~ C(store_location)', data=df).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

alpha = 0.05
if anova_table['PR(>F)'][0] < alpha:
    print("Reject the null hypothesis. There is a significant difference in average sales revenue across store locations.")
    # Performing post-hoc test (Tukey's HSD) to identify which store locations differ significantly
    from statsmodels.sandbox.stats.multicomp import MultiComparison
    m_comp = MultiComparison(df['Total_Price'], df['store_location'])
    result = m_comp.tukeyhsd(alpha=0.05)
    print(result)
else:
    print("Fail to reject the null hypothesis. There is no significant difference in average sales revenue across store locations.")
```
![image](https://github.com/user-attachments/assets/acd44f6d-bed1-4eb2-aa5a-8a108db11b6b)

2). SQL
A). Sales Performance

- What are the peak sales hours, and how do they vary by day of the week?
``` sql
SELECT
FORMAT_TIMESTAMP('%A', transaction_date) AS weekday,
EXTRACT(HOUR FROM transaction_time) AS hour,
SUM(transaction_qty * unit_price) AS total_sales
FROM coffee_store.Sales
GROUP BY weekday, hour
ORDER BY total_sales DESC;
```
Results:
![image](https://github.com/user-attachments/assets/cec9e783-de07-4b0f-ac9b-a3b85bf51c71)

- 	Which products generate the highest and lowest sales revenue?
``` sql
SELECT
product_type,
SUM(transaction_qty * unit_price) AS total_revenue
FROM coffee_store.Sales
GROUP BY product_type
ORDER BY total_revenue DESC
LIMIT 5;
```
Results:
![image](https://github.com/user-attachments/assets/eecac50e-d8fd-46a3-a100-1a6ed8141612)
- Highest Sales:
![image](https://github.com/user-attachments/assets/32e294fb-dcc0-4976-8e3c-a8faf01e7cbd)
- Lowest Sales
![image](https://github.com/user-attachments/assets/cc50b03b-44cb-4d32-8c83-1a524b045526)

- Are there seasonal trends affecting sales performance?
``` sql
SELECT
EXTRACT(YEAR FROM transaction_date) AS year,
EXTRACT(MONTH FROM transaction_date) AS month,
SUM(transaction_qty * unit_price) AS total_sales
FROM coffee_store.Sales
GROUP BY year, month
ORDER BY year, month;
```
Results:
![image](https://github.com/user-attachments/assets/3e2938b2-27a1-47ef-8338-c2b786f7f4eb)

-	Do weekends have significantly higher sales compared to weekdays?
``` sql
SELECT
CASE
WHEN FORMAT_TIMESTAMP('%A', transaction_date) IN ('Saturday', 'Sunday') THEN 'Weekend' ELSE 'Weekday'
END AS day_type,
SUM(transaction_qty * unit_price) AS total_sales
FROM coffee_store.Sales
GROUP BY day_type
ORDER BY total_sales DESC;
```
Results:
![image](https://github.com/user-attachments/assets/17afe0c2-5aa9-46c6-92f1-124faa20b4d8)

B). Pricing Strategy & Revenue Optimization
-	How does unit price affect the quantity sold?
``` sql
SELECT
CORR(unit_price, transaction_qty) AS correlation
FROM coffee_store.Sales;
```
![image](https://github.com/user-attachments/assets/f74b1c69-e1d0-4ce0-858b-132c735258c3)
-In our case, there is a Negative correlation where it lead to fewer quantity sold.

-	What is the average price elasticity of demand for different products?
``` sql
WITH price_changes AS (
SELECT
transaction_id,
LAG(unit_price) OVER (PARTITION BY transaction_id ORDER BY transaction_date) AS prev_price,
unit_price AS current_price,
LAG(transaction_qty) OVER (PARTITION BY transaction_id ORDER BY transaction_date) AS prev_qty,
transaction_qty AS current_qty
FROM coffee_store.Sales
),
percentage_changes AS (
SELECT
transaction_id,
((current_qty - prev_qty) / NULLIF(prev_qty, 0)) AS quantity_change_pct,
((current_price - prev_price) / NULLIF(prev_price, 0)) AS price_change_pct
FROM price_changes
WHERE prev_price IS NOT NULL AND prev_qty IS NOT NULL
)
SELECT
transaction_id,
AVG(quantity_change_pct / NULLIF(price_change_pct, 0)) AS avg_price_elasticity
FROM percentage_changes
GROUP BY transaction_id
ORDER BY avg_price_elasticity DESC;
```
Results:(Refer phython)

- Are premium-priced products underperforming compared to budget-friendly items?
``` sql
WITH product_price_segments AS (
SELECT
product_id,
unit_price,
NTILE(3) OVER (ORDER BY unit_price) AS price_segment
FROM (
SELECT product_id, AVG(unit_price) AS unit_price
FROM coffee_store.Sales
GROUP BY product_id
)
),
sales_performance AS (
SELECT
pps.price_segment,
COUNT(DISTINCT s.product_id) AS total_products,
SUM(s.transaction_qty * s.unit_price) AS total_revenue,
SUM(s.transaction_qty) AS total_quantity_sold
FROM coffee_store.Sales s
JOIN product_price_segments pps ON s.product_id = pps.product_id
GROUP BY pps.price_segment
)
SELECT
    CASE
        WHEN price_segment = 1 THEN 'Budget-Friendly'
        WHEN price_segment = 2 THEN 'Mid-Range'
        WHEN price_segment = 3 THEN 'Premium'
    END AS price_category,
    total_products,
    total_revenue,
    total_quantity_sold
FROM sales_performance
ORDER BY total_revenue DESC;
```
Results:
![image](https://github.com/user-attachments/assets/93ed967a-1f4e-4964-bf07-54023ecb34d5)
-In our case, Budget Friendly products were sold more and customers are price sensitive.
Note:
-	If premium-priced products have lower total sales revenue or lower quantity sold, they might be underperforming.
-	If budget-friendly items dominate revenue, customers might be price-sensitive.
-	If premium products perform well, pricing strategy is effective.

C). Customer Behavior Analysis
-	Which product categories are most frequently purchased together? (Market Basket Analysis)
``` sql
WITH product_combinations AS (
SELECT
t1.transaction_id,
t1.product_category AS category_1,
t2.product_category AS category_2
FROM coffee_store.Sales t1
JOIN coffee_store.Sales t2
ON t1.transaction_id = t2.transaction_id
AND t1.product_category < t2.product_category
)
SELECT
category_1,
category_2,
COUNT(*) AS frequency,
COUNT(*) * 1.0 / (SELECT COUNT(DISTINCT transaction_id) FROM coffee_store.Sales) AS support
FROM product_combinations
GROUP BY category_1, category_2
ORDER BY frequency DESC
LIMIT 10;
```
Results: (refer python)

-	Are there specific time slots where customers buy more premium items?
``` sql
WITH PremiumItems AS (
SELECT
*,
CASE
WHEN unit_price > (SELECT AVG(unit_price) FROM coffee_store.Sales) THEN 'Premium'ELSE 'Regular'
END AS item_category
FROM coffee_store.Sales
)
SELECT
EXTRACT(HOUR FROM transaction_time) AS hour,
item_category,
COUNT(transaction_id) AS total_transactions,
SUM(transaction_qty * unit_price) AS total_sales
FROM PremiumItems
GROUP BY hour, item_category
ORDER BY hour, item_category;
```
Results:
![image](https://github.com/user-attachments/assets/8aa0a467-0d56-41f2-aecd-c9af54506729)

-	What is the average transaction size per customer?
``` sql
SELECT
transaction_id,
SUM(transaction_qty * unit_price) / COUNT(DISTINCT transaction_id) AS avg_transaction_size
FROM coffee_store.Sales
GROUP BY transaction_id
ORDER BY avg_transaction_size DESC;
```
Results:
![image](https://github.com/user-attachments/assets/30283f03-00d9-45a8-86f4-945a52a3b998)

D). Store Location Performance Analysis
-	Which store locations have the highest/lowest revenue?
``` sql
SELECT
store_location,
SUM(transaction_qty * unit_price) AS total_revenue
FROM coffee_store.Sales
GROUP BY store_location
ORDER BY total_revenue DESC;
```
Results:
![image](https://github.com/user-attachments/assets/a1717a47-fd27-4833-a197-a6f9fe930e6d)
In this case, Hell’s Kitchen has the highest revenue and Lower Manhattan has the lowest revenue.

- Do different stores have different customer purchase behaviors?
```sql
SELECT
    store_location,
    COUNT(DISTINCT transaction_id) AS total_transactions,
    SUM(transaction_id *transaction_qty) / COUNT(DISTINCT transaction_id) AS avg_transaction_value,
    SUM(transaction_qty) / COUNT(DISTINCT transaction_id) AS avg_items_per_transaction
FROM coffee_store.Sales
GROUP BY store_location
ORDER BY avg_transaction_value DESC;
```
Results:
![image](https://github.com/user-attachments/assets/434c2176-50d2-4448-8bba-09715b1bbce9)

E). Inventory & Waste Management
- Which products experience the highest stock wastage due to low demand?
Note: if we had total_stock_received then it would have helped with this formula : Stock wastage = total_stock_received - total_sold_quantity)
``` sql
WITH product_sales AS (
SELECT
product_category,
SUM(transaction_qty) AS total_sold
FROM coffee_store.Sales
GROUP BY product_category
),
low_demand_products AS (
SELECT
product_category,
total_sold
FROM product_sales
WHERE total_sold < (
SELECT APPROX_QUANTILES(total_sold, 4)[OFFSET(1)] AS Q1
FROM product_sales
    )
)
SELECT
ROW_NUMBER() OVER () AS rank,
product_category,
total_sold
FROM low_demand_products;
```
Results:
![image](https://github.com/user-attachments/assets/6511c986-9365-4838-9e3b-f02fd033341a)

- 	How much inventory should be stocked based on expected demand?
``` sql
WITH weekly_sales AS (
    SELECT
        product_category,
        EXTRACT(WEEK FROM transaction_date) AS week_number,
        SUM(transaction_qty) AS total_weekly_sales
    FROM coffee_store.Sales
    GROUP BY product_category, week_number
),
average_weekly_sales AS (
    SELECT
        product_category,
        AVG(total_weekly_sales) AS avg_sales_per_week
    FROM weekly_sales
    GROUP BY product_category
)
SELECT
    aws.product_category,
    (aws.avg_sales_per_week * 2) AS recommended_inventory -- 2 weeks safety stock
FROM average_weekly_sales aws
ORDER BY recommended_inventory DESC;
```
Results:
![image](https://github.com/user-attachments/assets/1aea6fed-967b-41e6-aff5-6960bdb5ce5d)
f). Marketing & Promotion Effectiveness
-	How do discounts and promotions affect product sales?
``` sql
WITH price_changes AS (
    SELECT
        product_id,
        transaction_date,
        unit_price,
        transaction_qty,  -- ✅ Add this column to be used in the final query
        LAG(unit_price) OVER (PARTITION BY product_id ORDER BY transaction_date) AS prev_price
    FROM coffee_store.Sales
)
SELECT
    product_id,
    COUNT(CASE WHEN prev_price > unit_price THEN 1 END) AS num_discount_events,
    SUM(CASE WHEN prev_price > unit_price THEN transaction_qty ELSE 0 END) AS total_sold_with_discount,
    SUM(transaction_qty) AS total_quantity_sold,
    (SUM(CASE WHEN prev_price > unit_price THEN transaction_qty ELSE 0 END) * 100.0 / NULLIF(SUM(transaction_qty), 0)) AS discount_sales_percentage
FROM price_changes
GROUP BY product_id
ORDER BY discount_sales_percentage DESC;
```
Results:
![image](https://github.com/user-attachments/assets/b40e0985-a07b-419b-9039-4bc3e9e9965d)

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
