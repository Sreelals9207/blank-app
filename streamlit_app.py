import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


st.title("Clustering of Retail shop customers.")
selected_type = st.radio("Choose data type:", ('Trimmed', 'Non-Trimmed'))

def scatter_plot_df(selected_type = "Trimmed"):
    data = pd.read_csv("/Users/user/Downloads/OnlineRetail.csv.zip", encoding='ISO-8859-1')

    #making the data into a dataframe
    df = pd.DataFrame(data)

    #implimenting a new column 'totalprize'
    df["totalprize"] = df["Quantity"] * df["UnitPrice"]

    #calculating the total money spend by each customer
    df1 = df.groupby("CustomerID")["totalprize"].sum().reset_index()

    #calculating the total quandity purchased by each customer
    df2 = df.groupby("CustomerID")["Quantity"].sum().reset_index()

    #combainning 'df1' and 'df2' in a single dataframe
    df = pd.merge(df1, df2, on='CustomerID', how='outer')

    #rename the column 'Quandity'
    df.rename(columns={"Quantity" : "totalquandity"}, inplace=True)
    
    if selected_type == 'Trimmed':
        df.set_index("CustomerID", inplace=True)
        df = df[["totalprize", "totalquandity"]]
    else:
        df.set_index("CustomerID", inplace=True)
                #trimmed 10% bottom and top of 'df'.
        lower_bound_col1 = df["totalprize"].quantile(0.02)
        upper_bound_col1 = df["totalprize"].quantile(0.98)
        
        lower_bound_col2 = df["totalquandity"].quantile(0.02)
        upper_bound_col2 = df["totalquandity"].quantile(0.98)
        
        df = df[
            (df["totalprize"] >= lower_bound_col1) & (df["totalprize"] <= upper_bound_col1) &
            (df["totalquandity"] >= lower_bound_col2) & (df["totalquandity"] <= upper_bound_col2)
        ]

      fig = px.scatter(df,x="totalprize",y="totalquandity",labels={"totalprize": "Total Prize Given by Customer","totalquandity": "Total Quantity Purchased by Customers"},title="Distribution of Total Price and Total Quantity Purchased")
        return fig

fig = scatter_plot_df(selected_type)
st.plotly_chart(fig)

st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
