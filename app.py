import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Global Business Anlaytics and Sales Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)


def assess_trend_for_item(df: pd.DataFrame, threshold: float = 0.10) -> str:
    df["sales_next_year"] = df["total_sales"].shift(-12)
    df_compare = df.dropna(subset=["sales_next_year"])
    if df_compare.empty:
        return "not enough data"
    epsilon = 1e-6
    change_pct = (df_compare["sales_next_year"] - df_compare["total_sales"]) / (
        df_compare["total_sales"] + epsilon
    )
    def classify_change(pct_change):
        if pct_change > threshold:
            return "rising"
        elif pct_change < -threshold:
            return "falling"
        else:
            return "stable"  # Within +/- threshold
    df_compare["trend_class"] = change_pct.apply(classify_change)
    class_counts = df_compare["trend_class"].value_counts()
    total_comparisons = len(df_compare)
    if class_counts.empty:
        return "not enough data"
    most_frequent_class = class_counts.index[0]
    frequency_of_mode = class_counts.iloc[0]
    if frequency_of_mode / total_comparisons < 0.5:
        # Check for near tie between top two
        if len(class_counts) > 1 and (class_counts.iloc[0] - class_counts.iloc[1] <= 1):
            return "unstable"
    return most_frequent_class

def analyze_sales_trends(data_dict: dict) -> dict:
    results = {}
    for item_key, item_df in data_dict.items():
        item_df["total_sales"] = pd.to_numeric(item_df["total_sales"], errors="coerce")
        trend = assess_trend_for_item(item_df)
        results[item_key] = trend
    return results

@st.cache_data
def load_data(csv_path, json_path):
    """Loads, concatenates, and preprocesses the sales data."""
    try:
        csv_file = pd.read_csv(csv_path)
        json_data = pd.read_json(json_path)
        df = pd.concat([csv_file, json_data], ignore_index=True)

        df = df.drop(columns=[
            "ADDRESSLINE1", "ADDRESSLINE2", "STATE", "TERRITORY", "POSTALCODE",
            "ORDERLINENUMBER", "QTR_ID", "MSRP", "PHONE", "CONTACTLASTNAME",
            "CONTACTFIRSTNAME"
        ])

        df.dropna(subset=['CUSTOMERNAME', 'SALES', 'ORDERDATE'], inplace=True)

        order_date_series = pd.Series([str(el).split(" ")[0] for el in df["ORDERDATE"]])
        df["ORDERDATE"] = pd.to_datetime(order_date_series)

        df['year_month'] = df['ORDERDATE'].dt.to_period('M')

        all_year_months = df['year_month'].unique()
        countries = df['COUNTRY'].unique()
        countries_performance_record = {}
        for country in countries:
            country_data = df[df['COUNTRY'] == country].groupby('year_month').agg(
                total_sales=("SALES", 'sum')
            ).reset_index()
            complete_data = pd.DataFrame({'year_month': all_year_months})
            complete_data = complete_data.merge(country_data, on='year_month', how='left')
            complete_data['total_sales'] = complete_data['total_sales'].fillna(0)
            complete_data = complete_data.sort_values('year_month').reset_index(drop=True)
            countries_performance_record[country] = complete_data

        countries_trend_assessment_record = {}
        results = analyze_sales_trends(countries_performance_record)
        for key, value in results.items():
            countries_trend_assessment_record.setdefault(value, []).append(key)

        products = df['PRODUCTCODE'].unique()
        products_performance_record = {}
        for product in products:
            product_data = df[df['PRODUCTCODE'] == product].groupby('year_month').agg(
                total_sales=('SALES', 'sum')
            ).reset_index()
            complete_data = pd.DataFrame({'year_month': all_year_months})
            complete_data = complete_data.merge(product_data, on='year_month', how='left')
            complete_data['total_sales'] = complete_data['total_sales'].fillna(0)
            complete_data = complete_data.sort_values('year_month').reset_index(drop=True)
            products_performance_record[product] = complete_data

        products_performance_assessment_record = {}
        results = analyze_sales_trends(products_performance_record)
        for key, value in results.items():
            products_performance_assessment_record.setdefault(value, []).append(key)

        training_data = df[["ORDERDATE", "QUANTITYORDERED", "SALES"]].copy()
        training_data = training_data.groupby('ORDERDATE').agg(
            orders_quantity=('QUANTITYORDERED', 'sum'),
            total_sales=('SALES', 'sum')
        )

        return df, countries_trend_assessment_record, products_performance_assessment_record, training_data

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.stop()


DF, COUNTRIES_TRENDS, PRODUCTS_TRENDS, TRAINING_DATA = load_data(
    csv_path="./data/sales_data_sample.csv",
    json_path="./data/sales_data_sample.json"
)

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    st.header("Sales Data Overview")
    st.write(f"Total Records: **{len(DF):,}**")
    st.write(f"Timeframe: **{DF['ORDERDATE'].min().strftime('%b %Y')}** - **{DF['ORDERDATE'].max().strftime('%b %Y')}**")
    st.write(f"Total Countries: **{DF['COUNTRY'].nunique()}**")
    st.write(f"Total Sales: **${DF['SALES'].sum():,.2f}**")

    st.markdown("---")

    st.subheader("Trend Analysis Settings")
    trend_threshold = st.slider("Select Sales Change Threshold (%)",min_value=1, max_value=20, value=10, step=1)
    st.write(f"Current Threshold: $\pm$**{trend_threshold}%**")


st.title("📈 Global Business Anlaytics and Sales Forecasting")
st.markdown("An interactive dashboard for analyzing sales performance and predicting future sales.")

st.header("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Sales", value=f"${DF['SALES'].sum():,.0f}")
with col2:
    st.metric(label="Total Orders", value=f"{DF['ORDERNUMBER'].nunique():,}")
with col3:
    avg_order_value = DF['SALES'].sum() / DF['ORDERNUMBER'].nunique()
    st.metric(label="Avg. Order Value", value=f"${avg_order_value:,.2f}")
with col4:
    st.metric(label="Total Clients", value=f"{DF['CUSTOMERNAME'].nunique():,}")

st.markdown("---")

st.header("🌎 Sales Distribution")
col_country, col_month = st.columns(2)

country_aggregated_data = DF.groupby('COUNTRY').agg(
    total_sales=('SALES', 'sum'), 
    clients_count=('CUSTOMERNAME', 'count')
).sort_values('total_sales', ascending=False).reset_index()

with col_country:
    st.subheader("Sales by Country")
    
    fig_country = px.bar(
        country_aggregated_data, 
        x='COUNTRY', 
        y='total_sales', 
        title='Total Sales by Country',
        color='total_sales',
        color_continuous_scale=px.colors.sequential.Plasma,
        template='seaborn'
    )
    fig_country.update_layout(
        xaxis={'categoryorder':'total descending'}, 
        xaxis_title="Country", 
        yaxis_title="Total Sales ($)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_country, use_container_width=True)
    

month_id_aggregated_data = DF.groupby(['MONTH_ID']).agg(
    total_sales=('SALES', 'sum'),
    clients_count=('CUSTOMERNAME', 'count')
).reset_index().sort_values('MONTH_ID')

with col_month:
    st.subheader("Sales by Month (Aggregated)")
    
    fig_month = px.line(
        month_id_aggregated_data, 
        x='MONTH_ID', 
        y='total_sales', 
        title='Total Sales by Month ID (1-12)',
        markers=True,
        template='seaborn'
    )
    fig_month.update_traces(line=dict(width=4))
    fig_month.update_layout(
        xaxis_title="Month ID", 
        yaxis_title="Total Sales ($)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_month, use_container_width=True)

with st.expander("Top and Bottom Performing Countries"):
    st.markdown("**Top 3 Countries by Sales**:")
    st.dataframe(country_aggregated_data[['COUNTRY', 'total_sales']].head(3), hide_index=True)
    st.markdown("**Bottom 3 Countries by Sales**:")
    st.dataframe(country_aggregated_data[['COUNTRY', 'total_sales']].tail(3), hide_index=True)

st.markdown("---")

st.header("📦 Product Sales Performance")

products_performance = DF.groupby("PRODUCTCODE").agg(
    total_product_sales=("SALES", "sum")
).sort_values('total_product_sales', ascending=False).reset_index()

top_n = st.slider("Select Top N Products to Display", min_value=5, max_value=len(products_performance), value=15)
filtered_products_performance = products_performance.head(top_n)

fig_product = px.bar(
    filtered_products_performance, 
    x='PRODUCTCODE', 
    y='total_product_sales', 
    title=f'Total Sales by Top {top_n} Product Codes',
    color='total_product_sales',
    color_continuous_scale=px.colors.sequential.Sunset,
    template='seaborn'
)
fig_product.update_layout(
    xaxis={'categoryorder':'total descending'}, 
    xaxis_title="Product Code", 
    yaxis_title="Total Sales ($)",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_product, use_container_width=True)

st.markdown("---")

st.header("💰 Deal Size Analysis")
deal_sizes = pd.crosstab(DF['COUNTRY'], DF['DEALSIZE']).reset_index()
deal_sizes_df = deal_sizes.set_index('COUNTRY')
deal_sizes_df = deal_sizes_df.reindex(columns=['Small', 'Medium', 'Large']) 

fig_deal_size = px.bar(
    deal_sizes_df, 
    x=deal_sizes_df.index, 
    y=['Small', 'Medium', 'Large'], 
    title='Deal Size Distribution by Country',
    labels={'x': 'Country', 'value': 'Number of Deals', 'variable': 'Deal Size'},
    color_discrete_map={'Small': '#4CAF50', 'Medium': '#FFC107', 'Large': '#F44336'},
    template='seaborn'
)
fig_deal_size.update_layout(
    xaxis_tickangle=-45,
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_deal_size, use_container_width=True)

with st.expander("Dominant Deal Size Classification"):
    dominant_deal_size = deal_sizes_df[['Large', 'Medium', 'Small']].idxmax(axis=1)
    countries_classified = dominant_deal_size.groupby(dominant_deal_size).groups
    
    for key, countries_list in countries_classified.items():
        st.markdown(f"**{key} Dominant Countries** ({len(countries_list)}): {', '.join(countries_list)}")

st.markdown("---")

st.header("⏱️ Year-over-Year Trend Summary")
st.info("Trend is assessed by comparing sales in each 12-month period to the following 12-month period. (Default threshold: $\pm$10%)")

col_trends_country, col_trends_product = st.columns(2)

with col_trends_country:
    st.subheader("Country Trends")
    for trend, countries_list in COUNTRIES_TRENDS.items():
        count = len(countries_list)
        if trend == 'rising':
            emoji = "⬆️"
            color = "success"
        elif trend == 'falling':
            emoji = "⬇️"
            color = "error"
        else:
            emoji = "➡️"
            color = "info"
            
        with st.expander(f"**{emoji} {trend.upper()}** ({count} countries)", expanded=False):
            if color == "success":
                st.success(f"**Countries:** {', '.join(countries_list)}")
            elif color == "error":
                st.error(f"**Countries:** {', '.join(countries_list)}")
            else:
                st.info(f"**Countries:** {', '.join(countries_list)}")

with col_trends_product:
    st.subheader("Product Trends")
    for trend, products_list in PRODUCTS_TRENDS.items():
        count = len(products_list)
        if trend == 'rising':
            emoji = "⬆️"
            color = "success"
        elif trend == 'falling':
            emoji = "⬇️"
            color = "error"
        else:
            emoji = "➡️"
            color = "info"
            
        with st.expander(f"**{emoji} {trend.upper()}** ({count} products)", expanded=False):
            if color == "success":
                st.success(f"**Products:** {', '.join(products_list)}")
            elif color == "error":
                st.error(f"**Products:** {', '.join(products_list)}")
            else:
                st.info(f"**Products:** {', '.join(products_list)}")

st.markdown("---")

st.header("🔮 Sales Forecasting (Prophet Model)")

@st.cache_resource
def train_prophet_model(training_data):
    train_subset = training_data.loc[training_data.index < pd.to_datetime("2005-01-01")].copy()
    test_subset = training_data.loc[training_data.index >= pd.to_datetime("2005-01-01")].copy()

    Pmodel_train_subset = train_subset.reset_index().rename(columns={'ORDERDATE':'ds', 'total_sales':'y'})
    Pmodel_train_subset['orders_quantity'] = train_subset['orders_quantity'].values

    model = Prophet()
    model.add_regressor("orders_quantity")
    model.fit(Pmodel_train_subset)

    Pmodel_test_subset = test_subset.reset_index().rename(columns={'ORDERDATE':'ds', 'total_sales':'y'})
    Pmodel_test_subset['orders_quantity'] = test_subset['orders_quantity'].values
    
    forecasting_result = model.predict(Pmodel_test_subset)
    
    return Pmodel_test_subset, forecasting_result

with st.spinner("Training Sales Forecasting Model..."):
    TEST_DATA, FORECAST = train_prophet_model(TRAINING_DATA)

st.subheader("Actual vs. Forecasted Sales")

fig_forecast = px.line(
    x=TEST_DATA['ds'], 
    y=TEST_DATA['y'].values, 
    labels={'x': 'Date', 'y': 'Total Sales ($)'},
    title='Actual vs. Forecasted Sales',
    template='seaborn'
)

fig_forecast.add_scatter(
    x=TEST_DATA['ds'], 
    y=FORECAST['yhat'].values, 
    mode='lines', 
    name='Forecasted Sales', 
    line=dict(color='green', dash='dash')
)

fig_forecast.add_trace(
    go.Scatter(
        x=TEST_DATA['ds'],
        y=FORECAST['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(width=0),
        name='Upper Bound'
    )
)
fig_forecast.add_trace(
    go.Scatter(
        x=TEST_DATA['ds'],
        y=FORECAST['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0,128,0,0.2)',
        name='Confidence Interval'
    )
)

fig_forecast = go.Figure(data=fig_forecast.data, layout=fig_forecast.layout)

fig_forecast.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Sales ($)",
    hovermode="x unified"
)
st.plotly_chart(fig_forecast, use_container_width=True)

mape = np.mean(np.abs((TEST_DATA['y'].values - FORECAST['yhat'].values) / TEST_DATA['y'].values)) * 100
st.markdown(f"Model Performance (Test Set): **Mean Absolute Percentage Error (MAPE): {mape:,.2f}%**")