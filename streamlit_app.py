import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



st.title("Clustering of Retail shop customers.")
selected_type = st.radio("Choose data type:", ('Trimmed', 'Non-Trimmed'))

def scatter_plot_df(selected_type = "Trimmed"):
    data = pd.read_csv("OnlineRetail.csv.zip", encoding='ISO-8859-1')

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
    
    if selected_type == 'Non-Trimmed':
        df.set_index("CustomerID", inplace=True)
        df1 = df[["totalprize", "totalquandity"]]
    else:
        df.set_index("CustomerID", inplace=True)
                #trimmed 10% bottom and top of 'df'.
        lower_bound_col1 = df["totalprize"].quantile(0.02)
        upper_bound_col1 = df["totalprize"].quantile(0.98)
        
        lower_bound_col2 = df["totalquandity"].quantile(0.02)
        upper_bound_col2 = df["totalquandity"].quantile(0.98)
        
        df1 = df[
            (df["totalprize"] >= lower_bound_col1) & (df["totalprize"] <= upper_bound_col1) &
            (df["totalquandity"] >= lower_bound_col2) & (df["totalquandity"] <= upper_bound_col2)
        ]

    fig = px.scatter(
        df1,
        x="totalprize",
        y="totalquandity",
        labels={
            "totalprize": "Total Prize Given by Customer",
            "totalquandity": "Total Quantity Purchased by Customers"
        },
        title="Distribution of Total Price and Total Quantity Purchased"
    )
      
    shape = df1.shape
    return fig, shape
    

fig , shape = scatter_plot_df(selected_type)
st.plotly_chart(fig)
st.header(
    f"shape of the dataframe is{shape}."
)

def for_df(selected_type = "Trimmed"):
    data = pd.read_csv("OnlineRetail.csv.zip", encoding='ISO-8859-1')

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
    
    if selected_type == 'Non-Trimmed':
        df.set_index("CustomerID", inplace=True)
        df1 = df[["totalprize", "totalquandity"]]
    else:
        df.set_index("CustomerID", inplace=True)
                #trimmed 10% bottom and top of 'df'.
        lower_bound_col1 = df["totalprize"].quantile(0.02)
        upper_bound_col1 = df["totalprize"].quantile(0.98)
        
        lower_bound_col2 = df["totalquandity"].quantile(0.02)
        upper_bound_col2 = df["totalquandity"].quantile(0.98)
        
        df1 = df[
            (df["totalprize"] >= lower_bound_col1) & (df["totalprize"] <= upper_bound_col1) &
            (df["totalquandity"] >= lower_bound_col2) & (df["totalquandity"] <= upper_bound_col2)
        ]
    return df1

slider = st.slider("number of clusters for trimmed data", 2, 15, 2)
def k_models(selected_type = "Trimmed", slider=2):
    df = for_df(selected_type = selected_type)
    model = KMeans(n_clusters=slider, random_state=42)
    model.fit(df)
    inertia = model.inertia_
    ss = silhouette_score(df, model.labels_)
    return inertia, ss

inertia, ss = k_models(selected_type, slider)
st.subheader(f"inertia for the model : {inertia}.")
st.subheader(f"silhouette_score for the model : {ss}.") 




def scatter_with_cluster(selected_type="Trimmed", slider=2):
    df = for_df(selected_type=selected_type)  # Assuming for_df is defined elsewhere
    model = KMeans(n_clusters=slider, random_state=42)
    model.fit(df)
    labels = model.labels_
    centroids = model.cluster_centers_

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis
    sns.scatterplot(
        ax=ax,
        x=df["totalprize"],
        y=df["totalquandity"],
        hue=labels,
        palette="deep"
    )
    
    ax.scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        marker="*",
        s=340,
        color='red',  # You can customize the centroid color
        label='Centroids'
    )
    
    ax.set_xlabel("Total Prize Given by Customer")
    ax.set_ylabel("Total Quantity Purchased by Customers")
    ax.set_title("Distribution of Total Price and Total Quantity Purchased")
    ax.legend()

    return fig

figure = scatter_with_cluster(selected_type, slider)
st.pyplot(figure)





















