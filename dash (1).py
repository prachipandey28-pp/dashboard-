# -----------------------------------------------
# SUPERSTORE PREMIUM DASHBOARD (SINGLE FILE)
# PART 1 — BASE ENGINE, IMPORTS, THEMES, HELPERS
# -----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from fpdf import FPDF
import base64
import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Superstore Premium Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# THEME ENGINE (DARK / LIGHT)
# --------------------------------------------------
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #111111 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# KPI ANIMATION FUNCTION
# --------------------------------------------------
def animate_value(value, prefix=""):
    """
    Simple streamlit KPI animation function.
    """
    placeholder = st.empty()
    for i in range(0, int(value), max(1, int(value/50))):
        placeholder.metric("", f"{prefix}{i:,}")
    placeholder.metric("", f"{prefix}{value:,}")

# --------------------------------------------------
# PDF EXPORT CLASS (PROFESSIONAL 5–7 PAGE REPORT)
# --------------------------------------------------
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Superstore Analytics Report", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

pdf = PDFReport()

def export_pdf(text_lines):
    """
    Convert insights + charts into a downloadable PDF.
    """
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text_lines:
        pdf.multi_cell(0, 10, line)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    b64 = base64.b64encode(pdf_output.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Report.pdf">📄 Download PDF Report</a>'
    return href

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("SuperStoreOrders.xlsx")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["ship_date"] = pd.to_datetime(df["ship_date"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day

    # Forecast preparation
    df_forecast = df.groupby("order_date")["sales"].sum().reset_index()
    df_forecast.columns = ["ds", "y"]

    return df, df_forecast

df, df_forecast = load_data()

# --------------------------------------------------
# PROPHECT FORECAST FUNCTION
# --------------------------------------------------
def generate_forecast(df_forecast, periods=90):
    model = Prophet()
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    fig = px.line(forecast, x="ds", y="yhat", title="Sales Forecast (Prophet)")
    return fig, forecast
# -----------------------------------------------
# PART 2 — SIDEBAR FILTERS + TABS + KPIs
# -----------------------------------------------

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔍 Filters")

years = st.sidebar.multiselect(
    "Select Year",
    options=sorted(df["year"].unique()),
    default=sorted(df["year"].unique())
)

categories = st.sidebar.multiselect(
    "Category",
    options=df["category"].unique(),
    default=df["category"].unique()
)

countries = st.sidebar.multiselect(
    "Country",
    options=df["country"].unique(),
    default=df["country"].unique()
)

filtered = df[
    (df["year"].isin(years))
    & (df["category"].isin(categories))
    & (df["country"].isin(countries))
]

# -----------------------------------------------
# TABS
# -----------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 Sales Analytics",
    "👥 Customer Insights",
    "📦 Product Performance",
    "🌍 Geo Analytics",
    "🤖 AI Insights",
    "🗂 Data Explorer"
])

# -----------------------------------------------
# ANIMATED KPI CARDS
# -----------------------------------------------
with tab1:
    st.title("📊 Overview Dashboard (Premium)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Total Sales")
        animate_value(int(filtered["sales"].sum()), prefix="$")

    with col2:
        st.subheader("Total Profit")
        animate_value(int(filtered["profit"].sum()), prefix="$")

    with col3:
        st.subheader("Total Orders")
        animate_value(int(filtered["order_id"].nunique()))

    with col4:
        st.subheader("Avg Order Value")
        animate_value(int(filtered["sales"].mean()), prefix="$")

# -----------------------------------------------
# CHART SWITCHER (Pie, Donut, Bar, Line, Radar)
# -----------------------------------------------
with tab1:
    st.subheader("📌 Chart Switcher")

    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Pie", "Donut", "Bar", "Line", "Radar"]
    )

    if chart_type == "Pie":
        fig = px.pie(filtered, names="category", values="sales", title="Category Sales Share")

    elif chart_type == "Donut":
        fig = px.pie(filtered, names="category", values="sales", hole=0.4, title="Donut View")

    elif chart_type == "Bar":
        fig = px.bar(
            filtered.groupby("sub_category")["sales"].sum().reset_index(),
            x="sub_category", y="sales", color="sales", title="Sales by Sub-Category"
        )

    elif chart_type == "Line":
        monthly = filtered.groupby(["year","month"])["sales"].sum().reset_index()
        fig = px.line(monthly, x="month", y="sales", color="year", title="Monthly Trend")

    elif chart_type == "Radar":
        radar_df = filtered.groupby("category")[["sales","profit","quantity"]].sum().reset_index()
        fig = go.Figure()
        for i, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row["sales"], row["profit"], row["quantity"]],
                theta=["Sales","Profit","Quantity"],
                fill="toself",
                name=row["category"]
            ))
        fig.update_layout(title="Category Performance Radar Chart")

    st.plotly_chart(fig, use_container_width=True)
# -----------------------------------------------
# PART 3 — ADVANCED OVERVIEW VISUALS
# -----------------------------------------------

with tab1:

    st.subheader("📈 Year-over-Year Sales Trend")

    yoy = filtered.groupby(["year", "month"])["sales"].sum().reset_index()

    fig_yoy = px.line(
        yoy,
        x="month",
        y="sales",
        color="year",
        markers=True,
        title="Year-over-Year Monthly Sales Comparison"
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

    # --------------------------------------------------
    # DRILLDOWN BAR CHART
    # --------------------------------------------------
    st.subheader("🔍 Drilldown — Category → Sub-Category → Product")

    level = st.selectbox("Select Drilldown Level", ["Category", "Sub-Category", "Product"])

    if level == "Category":
        drill = filtered.groupby("category")["sales"].sum().reset_index()
        fig_drill = px.bar(drill, x="category", y="sales", color="sales",
                           title="Sales by Category")

    elif level == "Sub-Category":
        drill = filtered.groupby("sub_category")["sales"].sum().reset_index()
        fig_drill = px.bar(drill, x="sub_category", y="sales", color="sales",
                           title="Sales by Sub-Category")

    else:
        drill = filtered.groupby("product_name")["sales"].sum().nlargest(20).reset_index()
        fig_drill = px.bar(drill, x="product_name", y="sales", color="sales",
                           title="Top 20 Products by Sales")

    st.plotly_chart(fig_drill, use_container_width=True)

    # --------------------------------------------------
    # CALENDAR HEATMAP (Daily Activity)
    # --------------------------------------------------
    st.subheader("📅 Daily Sales Calendar Heatmap")

    df_cal = filtered.groupby(["year", "month", "day"])["sales"].sum().reset_index()
    df_cal["date"] = pd.to_datetime(df_cal[["year", "month", "day"]])

    fig_cal = px.density_heatmap(
        df_cal,
        x="day",
        y="month",
        z="sales",
        color_continuous_scale="Turbo",
        title="Daily Sales Heatmap"
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # --------------------------------------------------
    # CATEGORY SUMMARY TABLE
    # --------------------------------------------------
    st.subheader("📦 Category Summary")

    cat_summary = filtered.groupby("category")[["sales", "profit", "quantity"]].sum()
    st.dataframe(cat_summary.style.highlight_max(color="lightgreen"), height=200)

    # --------------------------------------------------
    # TOP CITIES TABLE
    # --------------------------------------------------
    st.subheader("🏙 Top 10 Cities by Sales")

    top_state = (
        filtered.groupby("state")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(top_state, height=250)
# ---------------------------------------------------------
# PART 4 — ADVANCED SALES ANALYTICS
# ---------------------------------------------------------

with tab2:

    st.title("📈 Advanced Sales Analytics")

    # ----------------------------------------
    # CORRELATION HEATMAP
    # ----------------------------------------
    st.subheader("📉 Correlation Heatmap")

    corr_cols = ["sales", "profit", "discount", "quantity", "shipping_cost"]
    corr = filtered[corr_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Between Key Metrics"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ----------------------------------------
    # PROFIT vs SALES SCATTER
    # ----------------------------------------
    st.subheader("💰 Profit vs Sales (Segment-wise)")

    fig_scatter = px.scatter(
        filtered,
        x="sales",
        y="profit",
        color="segment",
        size="quantity",
        title="Profit vs Sales Bubble Chart",
        opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ----------------------------------------
    # DISCOUNT IMPACT CURVE
    # ----------------------------------------
    st.subheader("🔻 Discount Impact on Profit")

    discount_df = filtered.groupby("discount")[["profit"]].mean().reset_index()

    fig_disc = px.line(
        discount_df,
        x="discount",
        y="profit",
        markers=True,
        title="How Discount Affects Profit"
    )
    st.plotly_chart(fig_disc, use_container_width=True)

    # ----------------------------------------
    # WEEKDAY vs WEEKEND ANALYSIS
    # ----------------------------------------
    st.subheader("📅 Weekday vs Weekend Sales")

    df_week = filtered.copy()
    df_week["weekday"] = df_week["order_date"].dt.day_name()

    fig_week = px.bar(
        df_week.groupby("weekday")["sales"].sum().reset_index(),
        x="weekday",
        y="sales",
        color="sales",
        title="Sales by Weekday"
    )
    st.plotly_chart(fig_week, use_container_width=True)

    # ----------------------------------------
    # SHIPPING MODE ANALYSIS
    # ----------------------------------------
    st.subheader("✈ Shipping Mode Performance")

    fig_ship = px.box(
        filtered,
        x="ship_mode",
        y="sales",
        color="ship_mode",
        title="Sales Distribution Across Shipping Modes"
    )
    st.plotly_chart(fig_ship, use_container_width=True)

    # ----------------------------------------
    # PROPHET FORECAST SECTION
    # ----------------------------------------
    st.subheader("🔮 Sales Forecast (Prophet Model)")

    periods = st.slider("Forecast Days", 30, 365, 120)

    fig_forecast, forecast_data = generate_forecast(df_forecast, periods=periods)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.success("Forecast generated using Facebook Prophet 🚀")
# ---------------------------------------------------------
# PART 5 — CUSTOMER INSIGHTS + PRODUCT PERFORMANCE
# ---------------------------------------------------------

# =========================================================
# CUSTOMER INSIGHTS TAB
# =========================================================
with tab3:

    st.title("👥 Customer Insights (Premium)")

    # ----------------------------------------
    # HIGH VALUE CUSTOMERS TABLE
    # ----------------------------------------
    st.subheader("💎 High-Value Customers (Top 20)")
    high_value = (
        filtered.groupby("customer_name")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    st.dataframe(high_value, height=250)

    # ----------------------------------------
    # 3D CUSTOMER SEGMENTATION (SALES, PROFIT, QUANTITY)
    # ----------------------------------------
    st.subheader("🧠 Customer Segmentation — 3D Clustering")

    cluster_df = filtered.groupby("customer_name")[["sales","profit","quantity"]].sum().reset_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[["sales","profit","quantity"]])

    kmeans = KMeans(n_clusters=3, n_init="auto")
    cluster_df["cluster"] = kmeans.fit_predict(scaled)

    fig_cluster = px.scatter_3d(
        cluster_df,
        x="sales",
        y="profit",
        z="quantity",
        color="cluster",
        title="3D Customer Segmentation"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # ----------------------------------------
    # WORDCLOUD (PRODUCT NAMES)
    # ----------------------------------------
    st.subheader("☁ WordCloud — Popular Products")

    products_text = " ".join(filtered["product_name"].astype(str).tolist())

    wc = WordCloud(background_color="white", width=800, height=400).generate(products_text)

    fig_wc, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig_wc)


# =========================================================
# PRODUCT PERFORMANCE TAB
# =========================================================
with tab4:

    st.title("📦 Product Performance (Premium)")

    # ----------------------------------------
    # TREEMAP (Category → Sub-Category → Product)
    # ----------------------------------------
    st.subheader("🗂 Category → Sub-Category → Product Treemap")

    fig_tree = px.treemap(
        filtered,
        path=["category", "sub_category", "product_name"],
        values="sales",
        color="profit",
        color_continuous_scale="RdYlGn",
        title="Treemap of Product Hierarchy"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # ----------------------------------------
    # PREMIUM SANKEY (Category → Sub → Segment → Ship Mode)
    # ----------------------------------------
    st.subheader("🔀 Premium Sankey Flow Diagram")

    sankey_df = filtered[["category","sub_category","segment","ship_mode"]]

    # Assign integer codes
    labels = list(pd.unique(sankey_df.values.ravel()))
    label_index = {label: idx for idx, label in enumerate(labels)}

    source = sankey_df["category"].map(label_index)
    target = sankey_df["sub_category"].map(label_index)

    # Build flow links
    flows = (
        sankey_df.groupby(["category","sub_category"])
        .size()
        .reset_index(name="count")
    )

    fig_sankey = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=25, thickness=15),
            link=dict(
                source=flows["category"].map(label_index),
                target=flows["sub_category"].map(label_index),
                value=flows["count"]
            )
        )
    )
    fig_sankey.update_layout(title="Category → Subcategory Flow")
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ----------------------------------------
    # TOP LOSS-MAKING PRODUCTS
    # ----------------------------------------
    st.subheader("⚠️ Top Loss-Making Products")

    loss_df = (
        filtered.groupby("product_name")["profit"]
        .sum()
        .sort_values()
        .head(15)
        .reset_index()
    )
    st.dataframe(loss_df, height=280)

    # ----------------------------------------
    # PROFITABILITY MATRIX (Sales vs Discount)
    # ----------------------------------------
    st.subheader("📉 Profitability Matrix (Sales vs Discount)")

    fig_matrix = px.scatter(
        filtered,
        x="discount",
        y="sales",
        color="profit",
        size="quantity",
        title="Profitability Matrix — Impact of Discount",
        opacity=0.7,
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)
# ---------------------------------------------------------
# PART 6 — GEO ANALYTICS + AI INSIGHTS + EXPORT SECTION
# ---------------------------------------------------------

# =========================================================
# GEO ANALYTICS TAB
# =========================================================
with tab5:

    st.title("🌍 Geo Analytics (Global Sales Performance)")

    # ----------------------------------------
    # CHOROPLETH MAP
    # ----------------------------------------
    st.subheader("🗺 Country-wise Sales (Choropleth)")

    country_sales = filtered.groupby("country")["sales"].sum().reset_index()

    fig_geo = px.choropleth(
        country_sales,
        locations="country",
        locationmode="country names",
        color="sales",
        color_continuous_scale="Plasma",
        title="Sales Distribution Across Countries"
    )

    st.plotly_chart(fig_geo, use_container_width=True)

    # ----------------------------------------
    # BUBBLE MAP
    # ----------------------------------------
    st.subheader("🌐 Bubble Map — Sales Volume by State")

    bubble_df = filtered.groupby(["state","country"])["sales"].sum().reset_index()

    fig_bubble = px.scatter_geo(
        bubble_df,
        locations="country",
        locationmode="country names",
        size="sales",
        hover_name="state",
        color="sales",
        title="Bubble Map — State Level Sales"
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    # ----------------------------------------
    # REGION ANALYTICS
    # ----------------------------------------
    st.subheader("🧭 Region Performance")

    region_sales = filtered.groupby("region")["sales"].sum().reset_index()
    fig_region = px.pie(region_sales, names="region", values="sales", title="Sales by Region")
    st.plotly_chart(fig_region, use_container_width=True)


# =========================================================
# AI INSIGHTS TAB
# =========================================================
with tab6:

    st.title("🤖 AI Insights Generator")

    st.subheader("📌 Auto-Generated Insights Based on Your Data")

    insights = []

    # Dynamic insights:
    top_year = filtered.groupby("year")["sales"].sum().idxmax()
    insights.append(f"• Highest selling year was **{top_year}**, showing strong performance over time.")

    best_cat = filtered.groupby("category")["sales"].sum().idxmax()
    insights.append(f"• Best performing category is **{best_cat}**, contributing the most revenue.")

    loss_cat = filtered.groupby("category")["profit"].sum().idxmin()
    insights.append(f"• Weakest profit category is **{loss_cat}** — consider reducing discounting.")

    top_state = filtered.groupby("state")["sales"].sum().idxmax()
    insights.append(f"• state with highest demand: **{top_state}**.")

    avg_discount = round(filtered["discount"].mean() * 100, 2)
    insights.append(f"• Average discount offered is **{avg_discount}%**, directly influencing profit margins.")

    ship_best = filtered.groupby("ship_mode")["sales"].sum().idxmax()
    insights.append(f"• Most preferred shipping mode: **{ship_best}**.")

    for line in insights:
        st.write(line)

    st.success("AI insights generated successfully ✔")


# =========================================================
# PDF EXPORT (PROFESSIONAL REPORT)
# =========================================================
with tab6:

    st.subheader("📄 Export Full PDF Report")

    if st.button("Generate PDF"):
        pdf_text = [
            "Superstore Premium Report",
            "----------------------------------------",
            f"Total Sales: {filtered['sales'].sum():,.2f}",
            f"Total Profit: {filtered['profit'].sum():,.2f}",
            f"Best Category: {best_cat}",
            f"Weakest Category (Profit): {loss_cat}",
            f"Top state: {top_state}",
            "----------------------------------------",
            "AI Analytical Insights:",
        ]
        pdf_text.extend(insights)

        pdf_link = export_pdf(pdf_text)
        st.markdown(pdf_link, unsafe_allow_html=True)


# =========================================================
# DATA EXPLORER TAB
# =========================================================
with tab7:

    st.title("🗂 Data Explorer & Downloads")

    st.dataframe(filtered, use_container_width=True)

    # ----------------------------------------
    # DOWNLOAD CSV
    # ----------------------------------------
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv_data, "filtered_data.csv")

    # ----------------------------------------
    # DOWNLOAD EXCEL
    # ----------------------------------------
    excel_buffer = BytesIO()
    pd.ExcelWriter(excel_buffer, engine="xlsxwriter")
    filtered.to_excel(excel_buffer, index=False)
    st.download_button("⬇ Download Excel", excel_buffer.getvalue(), "filtered_data.xlsx")