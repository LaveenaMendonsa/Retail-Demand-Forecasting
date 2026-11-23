import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import date, datetime
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.title("📊 Retail Demand Forecasting & Analytics")
st.caption(
    "End-to-end Retail Sales Analysis, Machine Learning "
    "Model Comparison and Demand Forecasting"
)



# =====================================================
# 2. LIVE API DATA - ECUADOR PUBLIC HOLIDAYS
# =====================================================

@st.cache_data(ttl=21600)
def fetch_ecuador_holidays():
    """
    Fetch Ecuador public holidays from Nager.Date.
    Cache for 6 hours to avoid calling the API on every rerun.
    """
    today = date.today()
    years = [today.year, today.year + 1]
    holiday_frames = []

    for year in years:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/EC"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        records = response.json()

        if records:
            holiday_frames.append(pd.DataFrame(records))

    if not holiday_frames:
        return pd.DataFrame(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    holidays = pd.concat(holiday_frames, ignore_index=True)
    holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce")
    holidays = holidays.dropna(subset=["date"])
    holidays = holidays.sort_values("date").reset_index(drop=True)

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return holidays, fetched_at


def show_live_holiday_panel():
    st.sidebar.divider()
    st.sidebar.subheader("🌐 Live API: Ecuador Holidays")

    if st.sidebar.button("🔄 Refresh holiday data", use_container_width=True):
        fetch_ecuador_holidays.clear()

    try:
        holidays, fetched_at = fetch_ecuador_holidays()

        if holidays.empty:
            st.sidebar.warning("The API returned no holiday records.")
            return holidays

        today = pd.Timestamp(date.today())
        upcoming = holidays[holidays["date"] >= today].copy()

        st.sidebar.success("API connected")
        st.sidebar.caption(f"Last fetched: {fetched_at}")
        st.sidebar.caption("Source: Nager.Date public holidays API")

        if not upcoming.empty:
            next_holiday = upcoming.iloc[0]
            st.sidebar.write("**Next public holiday**")
            st.sidebar.write(
                f"{next_holiday['date'].strftime('%d %b %Y')} — "
                f"{next_holiday.get('localName', next_holiday.get('name', 'Holiday'))}"
            )
        else:
            st.sidebar.info("No upcoming holidays found in the returned period.")

        return holidays

    except requests.RequestException as error:
        st.sidebar.error("Holiday API is temporarily unavailable.")
        st.sidebar.caption(f"Connection detail: {error}")
        return pd.DataFrame()

    except (ValueError, KeyError, TypeError) as error:
        st.sidebar.error("Could not process the holiday API response.")
        st.sidebar.caption(str(error))
        return pd.DataFrame()



# =====================================================
# 2B. ADDITIONAL LIVE APIs
# =====================================================

@st.cache_data(ttl=3600)
def fetch_ecuador_weather():
    """Fetch current weather for selected Ecuadorian cities using Open-Meteo."""
    cities = {
        "Quito": (-0.1807, -78.4678),
        "Guayaquil": (-2.1700, -79.9224),
        "Cuenca": (-2.9001, -79.0059),
        "Manta": (-0.9677, -80.7089),
    }

    rows = []
    for city, (latitude, longitude) in cities.items():
        url = "https://api.open-meteo.com/v1/forecast"
        response = requests.get(
            url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=15,
        )
        response.raise_for_status()
        current = response.json().get("current", {})
        rows.append({
            "City": city,
            "Temperature (°C)": current.get("temperature_2m"),
            "Humidity (%)": current.get("relative_humidity_2m"),
            "Wind speed (km/h)": current.get("wind_speed_10m"),
            "Observed at": current.get("time"),
        })

    return pd.DataFrame(rows), datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@st.cache_data(ttl=21600)
def fetch_exchange_rates():
    """Fetch latest available USD exchange rates from Frankfurter."""
    url = "https://api.frankfurter.dev/v1/latest"
    response = requests.get(
        url,
        params={"base": "USD", "symbols": "INR,EUR,GBP"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    rates = payload.get("rates", {})
    rates_df = pd.DataFrame(
        [{"Currency": currency, "Rate per 1 USD": rate}
         for currency, rate in rates.items()]
    )
    return rates_df, payload.get("date"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =====================================================
# 2. LOAD DATASETS
# =====================================================

@st.cache_data
def load_data():

    train = pd.read_csv(DATA_DIR / "train.csv")
    stores = pd.read_csv(DATA_DIR / "stores.csv")

    train["date"] = pd.to_datetime(train["date"])

    train = train.merge(
        stores,
        on="store_nbr",
        how="left",
        validate="many_to_one"
    )

    return train


@st.cache_data
def load_forecast():

    forecast_path = BASE_DIR / "submission_histgradientboosting.csv"
    test_path = DATA_DIR / "test.csv"

    if not forecast_path.exists() or not test_path.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(forecast_path)
    test = pd.read_csv(test_path)

    test["date"] = pd.to_datetime(test["date"])

    forecast = test[
        ["id", "date", "store_nbr", "family"]
    ].merge(
        predictions[["id", "sales"]],
        on="id",
        how="inner",
        validate="one_to_one"
    )

    return forecast


# Load both datasets before using them
try:
    df = load_data()
    forecast = load_forecast()

except Exception as e:
    st.error(f"Error loading project data: {e}")
    st.stop()


# =====================================================
# 3. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "Sales Analysis",
        "Product Analysis",
        "Store Analysis",
        "Model Comparison",
        "Demand Forecasting",
        "Live API Insights"
    ]
)

# Fetch live holiday context for the sidebar and Overview page
live_holidays = show_live_holiday_panel()


# =====================================================
# 4. HISTORICAL SALES DATE FILTER
# =====================================================

st.sidebar.divider()
st.sidebar.subheader("📅 Historical Sales Date Filter")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select historical date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="historical_dates"
)

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

else:
    start_date = selected_dates[0] if isinstance(
        selected_dates, (tuple, list)
    ) else selected_dates

    end_date = start_date


filtered_df = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()


# =====================================================
# 5. OVERVIEW
# =====================================================

if page == "Overview":

    st.header("Business Overview")

    total_sales = filtered_df["sales"].sum()
    total_stores = filtered_df["store_nbr"].nunique()
    total_families = filtered_df["family"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        "Total Stores",
        f"{total_stores}"
    )

    col3.metric(
        "Product Families",
        f"{total_families}"
    )

    st.divider()

    st.subheader("Daily Total Retail Sales")

    daily_sales = (
        filtered_df.groupby("date", as_index=False)["sales"]
        .sum()
    )

    fig = px.line(
        daily_sales,
        x="date",
        y="sales",
        title="Daily Total Retail Sales",
        labels={
            "date": "Date",
            "sales": "Total Sales"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales Contribution by Store Type")

    type_sales = (
        filtered_df.groupby("type", as_index=False)["sales"]
        .sum()
    )

    fig = px.pie(
        type_sales,
        names="type",
        values="sales",
        title="Sales by Store Type",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🌐 Live Ecuador Public Holidays")
    st.caption(
        "Automatically fetched from the Nager.Date API. "
        "This is live calendar context, not live sales data."
    )

    if not live_holidays.empty:
        today_ts = pd.Timestamp(date.today())
        upcoming_holidays = live_holidays[
            live_holidays["date"] >= today_ts
        ].copy()

        display_columns = [
            column for column in ["date", "localName", "name", "global"]
            if column in upcoming_holidays.columns
        ]

        st.dataframe(
            upcoming_holidays[display_columns].head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Live holiday information is unavailable right now. "
            "The rest of the dashboard can still be used."
        )


# =====================================================
# 6. SALES ANALYSIS
# =====================================================

elif page == "Live API Insights":

    st.header("🌐 Live External Data Insights")
    st.caption(
        "These APIs provide external business context. They do not replace "
        "the historical sales dataset or create new actual retail transactions."
    )

    st.subheader("🌦️ Current Weather in Selected Ecuadorian Cities")
    if st.button("Refresh weather data"):
        fetch_ecuador_weather.clear()

    try:
        weather_df, weather_fetched_at = fetch_ecuador_weather()
        st.success(f"Weather API connected | Retrieved: {weather_fetched_at}")
        st.dataframe(weather_df, use_container_width=True, hide_index=True)
    except requests.RequestException as error:
        st.warning(f"Weather API is temporarily unavailable: {error}")

    st.divider()
    st.subheader("💱 Latest Available Exchange Rates")
    st.caption(
        "Rates are quoted against USD. Frankfurter publishes reference rates; "
        "they may not be intraday market prices."
    )
    if st.button("Refresh exchange rates"):
        fetch_exchange_rates.clear()

    try:
        rates_df, rates_date, rates_fetched_at = fetch_exchange_rates()
        st.success(
            f"Exchange-rate API connected | Rate date: {rates_date} | "
            f"Retrieved: {rates_fetched_at}"
        )
        st.dataframe(rates_df, use_container_width=True, hide_index=True)
    except requests.RequestException as error:
        st.warning(f"Exchange-rate API is temporarily unavailable: {error}")

    st.divider()
    st.subheader("🎉 Ecuador Public Holidays")
    if not live_holidays.empty:
        today_ts = pd.Timestamp(date.today())
        upcoming = live_holidays[live_holidays["date"] >= today_ts].copy()
        holiday_columns = [
            col for col in ["date", "localName", "name", "global"]
            if col in upcoming.columns
        ]
        st.dataframe(
            upcoming[holiday_columns].head(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Holiday API data is currently unavailable.")


elif page == "Sales Analysis":

    st.header("📈 Sales Analysis")

    col1, col2 = st.columns(2)

    with col1:

        monthly_sales = (
            filtered_df.assign(
                month=filtered_df["date"].dt.to_period("M").astype(str)
            )
            .groupby("month", as_index=False)["sales"]
            .sum()
        )

        fig = px.bar(
            monthly_sales,
            x="month",
            y="sales",
            title="Monthly Sales Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        yearly_sales = (
            filtered_df.assign(
                year=filtered_df["date"].dt.year
            )
            .groupby("year", as_index=False)["sales"]
            .sum()
        )

        fig = px.bar(
            yearly_sales,
            x="year",
            y="sales",
            title="Yearly Sales Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Sales by Day of Week")

    weekday_sales = (
        filtered_df.assign(
            day=filtered_df["date"].dt.day_name()
        )
        .groupby("day", as_index=False)["sales"]
        .mean()
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_sales["day"] = pd.Categorical(
        weekday_sales["day"],
        categories=weekday_order,
        ordered=True
    )

    weekday_sales = weekday_sales.sort_values("day")

    fig = px.bar(
        weekday_sales,
        x="day",
        y="sales",
        title="Average Daily Sales"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 7. PRODUCT ANALYSIS
# =====================================================

elif page == "Product Analysis":

    st.header("🛒 Product Family Analysis")

    product_sales = (
        filtered_df.groupby("family", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )

    fig = px.bar(
        product_sales,
        x="family",
        y="sales",
        color="sales",
        title="Sales by Product Family"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Product Families")

    top_products = product_sales.head(10)

    fig = px.bar(
        top_products,
        x="sales",
        y="family",
        orientation="h",
        title="Top 10 Product Families"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Sales Details")

    st.dataframe(
        product_sales,
        use_container_width=True
    )


# =====================================================
# 8. STORE ANALYSIS
# =====================================================

elif page == "Store Analysis":

    st.header("🏬 Store Performance Analysis")

    store_sales = (
        filtered_df.groupby(
            ["store_nbr", "city", "state", "type"],
            as_index=False
        )["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            store_sales.head(15),
            x="store_nbr",
            y="sales",
            color="sales",
            title="Top 15 Stores by Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        city_sales = (
            filtered_df.groupby("city", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )

        fig = px.bar(
            city_sales.head(10),
            x="city",
            y="sales",
            title="Top 10 Cities by Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by Store Type")

    type_sales = (
        filtered_df.groupby("type", as_index=False)["sales"]
        .sum()
    )

    fig = px.pie(
        type_sales,
        names="type",
        values="sales",
        title="Store Type Sales Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Store Performance Table")

    st.dataframe(
        store_sales,
        use_container_width=True
    )


# =====================================================
# 9. MACHINE LEARNING MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.header("🤖 Machine Learning Model Comparison")

    st.write(
        "Validation performance of the trained regression models."
    )

    model_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "HistGradientBoosting"
        ],
        "MAE": [
            109.2781,
            134.140447
        ],
        "RMSE": [
            374.7767,
            333.925357
        ],
        "R2 Score": [
            0.9098720,
            0.9284494
        ]
    })

    st.subheader("Validation Metrics")

    st.dataframe(
        model_results.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Best MAE", "Random Forest")
    col2.metric("Best RMSE", "HistGradientBoosting")
    col3.metric("Best R²", "HistGradientBoosting")

    st.subheader("Compare Model Metrics")

    metric = st.selectbox(
        "Select evaluation metric",
        ["MAE", "RMSE", "R2 Score"]
    )

    fig = px.bar(
        model_results,
        x="Model",
        y=metric,
        color="Model",
        title=f"Model Comparison - {metric}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Random Forest has lower MAE. HistGradientBoosting "
        "has lower RMSE and higher R² on the validation data. "
        "R² is a regression score, not classification accuracy."
    )


# =====================================================
# 10. DEMAND FORECASTING
# =====================================================

elif page == "Demand Forecasting":

    st.header("🔮 Retail Demand Forecasting")

    st.write(
        "Predicted sales generated using the "
        "HistGradientBoosting model."
    )

    if forecast.empty:

        st.warning(
            "Forecast data is unavailable. Check that "
            "submission_histgradientboosting.csv exists "
            "in the project root folder."
        )

    else:

        forecast_min = forecast["date"].min().date()
        forecast_max = forecast["date"].max().date()

        st.subheader("Forecast Date Filter")

        forecast_dates = st.date_input(
            "Select forecast date range",
            value=(forecast_min, forecast_max),
            min_value=forecast_min,
            max_value=forecast_max,
            key="forecast_dates"
        )

        if isinstance(forecast_dates, (tuple, list)) and len(forecast_dates) == 2:

            forecast_start, forecast_end = forecast_dates

        else:

            forecast_start = forecast_min
            forecast_end = forecast_max

        forecast_filtered = forecast[
            (forecast["date"].dt.date >= forecast_start) &
            (forecast["date"].dt.date <= forecast_end)
        ].copy()

        if forecast_filtered.empty:

            st.warning(
                "No forecast records found for the selected dates."
            )

        else:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Forecast Records",
                f"{len(forecast_filtered):,}"
            )

            col2.metric(
                "Average Predicted Sales",
                f"{forecast_filtered['sales'].mean():,.2f}"
            )

            col3.metric(
                "Total Predicted Sales",
                f"{forecast_filtered['sales'].sum():,.2f}"
            )

            st.divider()

            st.subheader("Daily Predicted Retail Demand")

            daily_forecast = (
                forecast_filtered.groupby(
                    "date", as_index=False
                )["sales"]
                .sum()
            )

            fig = px.line(
                daily_forecast,
                x="date",
                y="sales",
                markers=True,
                title="Daily Predicted Retail Demand"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast by Product Family")

            family_forecast = (
                forecast_filtered.groupby(
                    "family", as_index=False
                )["sales"]
                .sum()
                .sort_values("sales", ascending=False)
            )

            fig = px.bar(
                family_forecast.head(10),
                x="family",
                y="sales",
                title="Top 10 Forecasted Product Families"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Data")

            st.dataframe(
                forecast_filtered.head(100),
                use_container_width=True
            )

            csv_data = forecast_filtered.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name="retail_demand_forecast.csv",
                mime="text/csv"
            )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "Retail Demand Forecasting & Analytics | "
    "Python | Pandas | Plotly | Streamlit | Machine Learning | REST APIs"
)
