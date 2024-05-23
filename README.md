# Sales-Prediction.

Introduction:

The dataset is provided  with historical sales data for 45 stores located in different regions - each store contains a number of departments. The company also runs several promotional markdown events throughout the year.
These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation 
than non-holiday weeks.
Within the Excel Sheet, there are 3 Tabs – Stores, Features and Sales

Explanation Of The Datasets:

Stores:    Anonymized information about the 45 stores, indicating the type and size of store.

Features:  Contains additional data related to the store, department, and regional activity for the given dates.
              
              Store - the store number
              Date - the week
              Temperature - average temperature in the region
              Fuel_Price - cost of fuel in the region
              MarkDown1-5 - anonymized data related to promotional markdowns. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA
              CPI - the consumer price index
              Unemployment - the unemployment rate
              IsHoliday - whether the week is a special holiday week

Sales:    Historical sales data, which covers to 2010-02-05 to 2012-11-01.

              Store - the store number
              Dept - the department number
              Date - the week
              Weekly_Sales -  sales for the given department in the given store
              IsHoliday - whether the week is a special holiday week

Goal of the Project:

- This dataset contains weekly sales from 81 departments belonging to 45 different stores. 

- The aim is to forecast weekly sales from a particular department.

- The objective of this case study is to forecast weekly retail store sales based on historical data.

- The data contains holidays and promotional markdowns offered by various stores and several departments throughout the year.

- Markdowns are crucial to promote sales especially before key events such as Super Bowl, Christmas and Thanksgiving. 

- Developing accurate model will enable make informed decisions and make recommendations to improve business processes in the future. 


