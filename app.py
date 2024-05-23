from datetime import date
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose

feature_df=pd.read_csv(r'C:\Users\ayish\GuviProjects\Final Project\Features_data_set (1).csv',parse_dates=['Date'])
store_df=pd.read_csv(r'C:\Users\ayish\GuviProjects\Final Project\stores_data_set.csv')
sales_df=pd.read_csv(r'C:\Users\ayish\GuviProjects\Final Project\sales_data_set (1).csv',parse_dates=['Date'])

feature_df['Date']=pd.to_datetime(feature_df['Date'],dayfirst=True,format='mixed')

feature_df['Day']=feature_df['Date'].dt.day
feature_df['month']=feature_df['Date'].dt.month
feature_df['year']=feature_df['Date'].dt.year

sales_df['Date']=pd.to_datetime(sales_df['Date'],dayfirst=True,format='mixed')

sales_df['Day']=sales_df['Date'].dt.day
sales_df['month']=sales_df['Date'].dt.month
sales_df['year']=sales_df['Date'].dt.year

data_date=feature_df.groupby('Date').agg({'Temperature':'mean',
                                          'Fuel_Price':'mean',
                                          'IsHoliday':'sum',
                                          'CPI':'mean',
                                          'Unemployment':'mean'
                                        })

data_date=data_date.sort_index()
#temp_date_data=data_date[:'2012-12-10']

data_sales_date=  sales_df.groupby("Date").agg({"Weekly_Sales":"sum"})
data_sales_date.sort_index(inplace=True)
data_sales_date.Weekly_Sales = data_sales_date.Weekly_Sales/1000000
data_sales_date.Weekly_Sales = data_sales_date.Weekly_Sales.apply(int)
data = pd.merge(data_sales_date,data_date, left_index=True,right_index=True, how='left')
data["IsHoliday"] = data["IsHoliday"].apply(lambda x: True if x == 45.0 else False )

def hotmap(df):
    fig1=sns.heatmap(df.corr(),annot=True)
    return st.pyplot(fig1.get_figure())


plt.style.use('fivethirtyeight')
fig, ax=plt.subplots(5,1, figsize=(10,5),sharex=True)
data['Weekly_Sales'].plot(ax=ax[0], title="Weekly sales/ sales on holiday")
data[data.IsHoliday==True]["Weekly_Sales"].plot(marker="D",ax=ax[0],legend="Holiday Week sale")
data["Temperature"].plot(ax=ax[1], title="Temperature")
data["Fuel_Price"].plot(ax=ax[2],title="Fuel_Price")
data["CPI"].plot(ax=ax[3],title="CPI")
data["Unemployment"].plot(ax=ax[4],title="Unemployment")



def streamlit_config():
    st.set_page_config(page_title='Weekly sales Prediction',layout='wide')
    page_background_color="""<style>[data-testid="stHeader"]{background: rgba(0,0,0,0);}</style>"""
    st.markdown(page_background_color, unsafe_allow_html=True)
    st.title(":violet[Weekly Sales Prediction]")
    #st.markdown(f'<h1 style="text-align: center;">Weekly sales Prediction</h1>',unsafe_allow_html=True)


def style_submit_button():
    st.markdown("""
                   <style>
                   div.stButton > button:first-child{
                   background-color: #367F89;
                   color: white;
                   width: 70%}
                   </style>
                """, unsafe_allow_html=True )

def style_prediction():
    st.markdown("""
                <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
            unsafe_allow_html=True
        )
class options:
    department=[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56,
       58, 59, 60, 67, 71, 72, 74, 77, 78, 79, 80, 81, 82, 83, 85, 87, 90,
       91, 92, 93, 94, 95, 96, 97, 98, 99, 39, 50, 43, 65]
    IsHoliday=[True, False]
    store=[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    
class prediction:
    def regression():
        with st.form('Regression'):
            store=st.selectbox(label='store', options=options.store)
            sales_date= st.date_input(label= 'date',min_value=date(2010,1,1), max_value=date(2030,5,31) ,value=date(2010,1,1))
            department=st.selectbox(label='department', options= options.department)
            IsHoliday=st.selectbox(label='Holiday', options=options.IsHoliday)

            st.write('')
            st.write('')
            button= st.form_submit_button(label='SUBMIT')
            style_submit_button()

        if button:
            with open(r'C:\Users\ayish\GuviProjects\Final Project\sales_model.pkl','rb') as f:
                model=pickle.load(f)

            user_data= np.array([[store,
                                  department,
                                  IsHoliday,
                                  sales_date.day, sales_date.month, sales_date.year]])
            
            y_pred=model.predict(user_data)
            weekly_sales= y_pred[0]
            return weekly_sales

streamlit_config()

show_table=st.radio("Select the option for view",("Analysis Datewise","Analysis storewise","Analysis Departmentwise","Analysis storetype", "Analysis of Markdown","Prediction"))
if show_table=="Analysis Datewise":

    col1, col2= st.columns(2)
    with col1:
        st.title('    Datewise Sales:')
        
    with col2:
        #st.markdown('##:green[**Nov- Dec shows spike in Weekly Sales.but over the year it is not increased.**]')
        #st.markdown('##:green[**Temperature is showing a random walk**]')
        #st.markdown('##:green[**weeks nearby holiday shows peak**]')
        #st.markdown('##:green[**Fuel Price and Consumer Price Index shown growth over the year.**]')
        #st.markdown('##:green[**Unemployment decreased year after year.**]')
        st.pyplot(fig)

     
    col1, col2= st.columns(2)
    with col1:
        st.markdown('')
        st.markdown('')
        st.title('   Month wise sales:')
    with col2:
        data_sales_month = data.groupby(data.index.month).agg({"Weekly_Sales":"sum"})
        plt.figure(figsize=(10, 5))
        barfig=sns.barplot(x=data_sales_month.index,y=data_sales_month.Weekly_Sales)
        plt.title("Month wise Sales")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        st.pyplot(barfig.get_figure())
    
        


    col1, col2= st.columns(2)

    with col1:
        st.title('Time series Analysis:')
    with col2:
        decomposition = seasonal_decompose(data["Weekly_Sales"], period=45)
        fig3=plt.figure(figsize=(15, 7))
        plt.plot(decomposition.trend)
        plt.plot(decomposition.seasonal)
        plt.plot(decomposition.resid)
        plt.legend(["Trend", "Seasonal","Resid"], loc ="upper right") 
        st.pyplot(fig3)

if show_table=="Analysis storewise":
    col1,col2=st.columns(2)
    with col1:
        st.title('Combining Count plot, Swarmplot, Box plot:')
    with col2:
        data_Store = feature_df.groupby("Store").agg({"Temperature":"mean","Fuel_Price":"mean","IsHoliday":"sum"})

        temp_store = sales_df.groupby("Store").agg({"Weekly_Sales":"sum"})
        temp_store.Weekly_Sales = temp_store.Weekly_Sales/1000000
        temp_store.Weekly_Sales = temp_store.Weekly_Sales.apply(int)
        data_Store.set_index(np.arange(0,45),inplace=True)
        store_df["temp"] = data_Store.Temperature
        store_df["Fuel_Price"] = data_Store.Fuel_Price
        store_df["holiday"] = data_Store.IsHoliday
        store_df["Weekly_Sales"] = temp_store.Weekly_Sales

        fig,ax = plt.subplots(1,3,figsize=(15, 10))
        sns.countplot(store_df.Type,ax=ax[0])
        sns.swarmplot(data = store_df,y="Size",x="Type",ax=ax[1])
        sns.boxplot(data = store_df,y="Weekly_Sales",x="Type",ax=ax[2])
        st.pyplot(fig)
if show_table=="Analysis Departmentwise":
    col1,col2=st.columns(2)
    with col1:
        st.title('Departmentwise sales:')
    with col2:
        data_Dept = sales_df.groupby("Dept").agg({"Weekly_Sales":"sum"})
        data_Dept.Weekly_Sales = data_Dept.Weekly_Sales/10000
        data_Dept.Weekly_Sales = data_Dept.Weekly_Sales.apply(int)
        data_Dept.sort_values(by="Weekly_Sales")
        fig1, ax1 = plt.subplots(figsize=(15, 10))

        plt.vlines(x=data_Dept.index, ymin=0, ymax=data_Dept['Weekly_Sales'], color='skyblue')
        plt.plot(data_Dept.index,data_Dept['Weekly_Sales'], "o")
        plt.title("Departmentwise Sales")
        plt.ylabel("Sales")
        plt.xlabel("Department")
        st.pyplot(fig1)

if show_table=="Analysis storetype":
    col1,col2=st.columns(2)
    with col1:
        st.title('Yearwise store type:')
    with col2:
        sales_date_store = sales_df.groupby(["Date","Store"]).agg({"Weekly_Sales":"sum"})
        sales_date_store.sort_index(inplace=True)
        sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales/10000
        sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales.apply(int)
        data_table = pd.merge(feature_df,sales_date_store ,  how='left', on=["Date","Store"])
        data_table = pd.merge(data_table,store_df[["Store","Type"]] ,  how='left', on=["Store"])
        data_table.head(20)
        data_train = data_table[data_table.Weekly_Sales.notnull()]
        data_test = data_table[data_table.Weekly_Sales.isnull()]
        fig=plt.figure(figsize=(15, 10))
        sns.barplot(x=data_train.Date.dt.year, y=data_train.Weekly_Sales,hue=data_train.Type)
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        st.title('Month wise store type:')
    with col2:
        fig=plt.figure(figsize=(15, 10))
        sns.barplot(x=data_train.Date.dt.month, y=data_train.Weekly_Sales,hue=data_train.Type)
        st.pyplot(fig)
if show_table=="Analysis of Markdown":
    col1,col2=st.columns(2)
    with col1:
        st.title('Timeline Markdown')
    with col2:
        sales_date_store = sales_df.groupby(["Date","Store"]).agg({"Weekly_Sales":"sum"})
        sales_date_store.sort_index(inplace=True)
        sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales/10000
        sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales.apply(int)
        data_table = pd.merge(feature_df,sales_date_store ,  how='left', on=["Date","Store"])
        data_table = pd.merge(data_table,store_df[["Store","Type"]] ,  how='left', on=["Store"])
        fig=plt.figure(figsize=(15,10))
        train_markdown = data_table[data_table.MarkDown2.notnull()]
        train_markdown = train_markdown.groupby("Date").agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})


        plt.plot(train_markdown.index,train_markdown.MarkDown1)
        plt.plot(train_markdown.index,train_markdown.MarkDown2)
        plt.plot(train_markdown.index,train_markdown.MarkDown3)
        plt.plot(train_markdown.index,train_markdown.MarkDown4)
        plt.plot(train_markdown.index,train_markdown.MarkDown5)
        plt.title("Timeline Markdown")
        plt.ylabel("Markdown")
        plt.xlabel("Date")
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        st.title('Markdown Differentiation:')
    with col2:
        fig=plt.figure(figsize=(15,10))
        plt.hist(train_markdown,bins=6, color=['yellow','green','orange','red','grey'])
        plt.tight_layout()
        plt.legend(["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"], loc ="upper right") 
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        st.title('Stacked Monthwise Markdown:')
    with col2:
        train_markdown_month = train_markdown.groupby(train_markdown.index.month).agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})
        #fig=plt.figure(figsize=(15,10))
        fig=px.bar(train_markdown_month)
        plt.title("Stacked Monthwise Morkdown")
        plt.ylabel("Markdown")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2=st.columns(2)
    with col1:
        st.title('Stacked Store Type wise:')
    with col2:
        train_markdown_1 = data_table[data_table.MarkDown2.notnull()]
        train_markdown_type = train_markdown_1.groupby("Type").agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})

        fig=px.bar(train_markdown_type)
        plt.title("Stacked StoreType Wise")
        plt.ylabel("Markdown")
        st.plotly_chart(fig)
        #tab1,tab2= st.tabs(['Predict weekly sales'])
#with tab1:

if show_table=='Prediction':

    st.title('Prediction Part:')
    try:
        weekly_sales=prediction.regression()
        if weekly_sales:
            style_prediction()
            st.markdown(f'### <div class="center-text">Predicted Selling Price = {weekly_sales}</div>', unsafe_allow_html=True)
    except ValueError:
        st.warning('##### department / holiday is empty')
