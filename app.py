# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from utils import load_data, summary_stats, predict_single
from sklearn.cluster import KMeans

# Page setup
st.set_page_config(layout="wide", page_title="Customer Churn Dashboard", initial_sidebar_state="expanded")

# Dataset and model paths
DATA_PATH = r"C:\Users\banuv\OneDrive\Desktop\customer Churn Analysis\customer_churn_dataset.csv"
MODEL_PATH = "model.joblib"

# Cached functions
@st.cache_data(show_spinner=False)
def load_dataset(path):
    return load_data(path)

@st.cache_resource(show_spinner=False)
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Retention advice
def retention_advice(row):
    if row["PredictedChurn"] == 1 and row["EngagementScore"] < 40:
        return "Send loyalty offers or engagement email"
    elif row["PredictedChurn"] == 1 and row["CustomerSatisfaction"] < 3:
        return "Customer support follow-up"
    elif row["PredictedChurn"] == 1 and row["Tenure"] < 6:
        return "Offer renewal discount or onboarding bonus"
    else:
        return "Stable - No action needed"

# ---------------- MAIN APP ----------------
def main():
    st.title("📊 Customer Churn Prediction & Sales Dashboard")
    st.caption("A clean analytical dashboard for customer churn analysis and prediction.")

    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at: {DATA_PATH}")
        return

    df = load_dataset(DATA_PATH)

    # Sidebar filters
    st.sidebar.header("Filters")
    region_sel = st.sidebar.multiselect("Region", df["Region"].unique().tolist(), default=df["Region"].unique().tolist())
    subscription_sel = st.sidebar.multiselect("Subscription Type", df["SubscriptionType"].unique().tolist(), default=df["SubscriptionType"].unique().tolist())
    churn_filter = st.sidebar.selectbox("Show", ["All", "Churned only", "Retained only"])
    engagement_range = st.sidebar.slider("Engagement Score", 0, 100, (0, 100))
    tenure_range = st.sidebar.slider("Tenure (months)", int(df["Tenure"].min()), int(df["Tenure"].max()),
                                     (int(df["Tenure"].min()), int(df["Tenure"].max())))

    # Apply filters
    filtered = df[
        (df["Region"].isin(region_sel))
        & (df["SubscriptionType"].isin(subscription_sel))
        & (df["EngagementScore"].between(*engagement_range))
        & (df["Tenure"].between(*tenure_range))
    ]
    if churn_filter == "Churned only":
        filtered = filtered[filtered["Churn"] == 1]
    elif churn_filter == "Retained only":
        filtered = filtered[filtered["Churn"] == 0]

    # KPI metrics
    kpis = summary_stats(filtered)
    full_kpis = summary_stats(df)
    delta_churn = (kpis["churn_rate"] - full_kpis["churn_rate"]) * 100

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", kpis["total_customers"])
    col2.metric("Churn Rate", f"{kpis['churn_rate']*100:.2f}%", f"{delta_churn:+.2f}% vs overall")
    col3.metric("Avg Monthly Charge", f"${kpis['avg_monthly']:.2f}")
    col4.metric("Avg Tenure (months)", f"{kpis['avg_tenure']:.1f}")

    st.markdown("---")

    # Visualization section
    colA, colB = st.columns((2, 3))
    with colA:
        st.subheader("Churn by Subscription Type")
        churn_by_sub = filtered.groupby("SubscriptionType")["Churn"].mean().reset_index()
        churn_by_sub["ChurnRate"] = churn_by_sub["Churn"] * 100
        fig1 = px.bar(churn_by_sub, x="SubscriptionType", y="ChurnRate",
                      text_auto=".2f", labels={"ChurnRate": "Churn Rate (%)"})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Churn by Region")
        churn_by_region = filtered.groupby("Region")["Churn"].mean().reset_index()
        churn_by_region["ChurnRate"] = churn_by_region["Churn"] * 100
        fig2 = px.pie(churn_by_region, names="Region", values="ChurnRate", title="Churn Rate by Region")
        st.plotly_chart(fig2, use_container_width=True)

    with colB:
        st.subheader("Engagement vs Monthly Charges")
        fig3 = px.scatter(filtered, x="MonthlyCharges", y="EngagementScore", color="Churn",
                          hover_data=["CustomerID", "SubscriptionType", "Region"])
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Engagement vs Satisfaction Heatmap")
        heat_data = pd.crosstab(filtered["EngagementScore"] // 10, filtered["CustomerSatisfaction"])
        fig_heat = px.imshow(heat_data, labels={"x": "Customer Satisfaction", "y": "Engagement (0–100)"},
                             color_continuous_scale="Greys")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = df.select_dtypes(include=["number"]).corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Greys")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Table section
    st.subheader("Filtered Customer Data")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered CSV", csv, "filtered_customers.csv", "text/csv")

    st.markdown("---")

    # Model predictions with retention advice
    st.subheader("Churn Predictions with Retention Advice")
    model = load_model(MODEL_PATH)
    if model is None:
        st.warning("Model not found. Run `python train_model.py` to train one.")
    else:
        st.success("Model Loaded Successfully!")
        selection = st.multiselect("Select Customer IDs (limit 50)",
                                   options=filtered["CustomerID"].tolist(),
                                   default=filtered["CustomerID"].tolist()[:5])
        if selection:
            subset = df[df["CustomerID"].isin(selection)].drop(columns=["CustomerID"])
            preds, probs = predict_single(model, subset)
            out = df[df["CustomerID"].isin(selection)].copy()
            out["PredictedChurn"] = preds
            out["ChurnProb"] = probs
            out["RetentionAdvice"] = out.apply(retention_advice, axis=1)

            st.dataframe(out[["CustomerID", "SubscriptionType", "Region", "EngagementScore",
                              "CustomerSatisfaction", "PredictedChurn", "ChurnProb", "RetentionAdvice"]])
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predictions_with_advice.csv", "text/csv")

    st.markdown("---")

    # What-if analysis
    st.subheader("What-if Churn Prediction Simulator")
    with st.form("what_if"):
        age = st.slider("Age", 18, 70, 30)
        tenure = st.slider("Tenure (months)", 1, 60, 10)
        engagement = st.slider("Engagement Score", 0, 100, 50)
        satisfaction = st.slider("Customer Satisfaction", 1, 5, 3)
        monthly = st.number_input("Monthly Charges", 10.0, 200.0, 50.0)
        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        payment = st.selectbox("Payment Method", ["Credit Card", "UPI", "PayPal", "Auto-Debit"])
        submitted = st.form_submit_button("Predict Churn")

        if submitted and model is not None:
            new_df = pd.DataFrame([{
                "Gender": "Male",
                "Age": age,
                "Region": "South",
                "Tenure": tenure,
                "SubscriptionType": subscription,
                "MonthlyCharges": monthly,
                "TotalSpend": monthly * tenure,
                "NumOfPurchases": np.random.randint(1, 100),
                "EngagementScore": engagement,
                "CustomerSatisfaction": satisfaction,
                "SupportTickets": np.random.randint(0, 5),
                "PaymentMethod": payment,
                "AutoRenewal": 1
            }])
            preds, probs = predict_single(model, new_df)
            st.success(f"Predicted Churn Probability: {probs[0]*100:.2f}%")
            if probs[0] > 0.5:
                st.warning("High Risk of Churn — Take retention action.")
            else:
                st.info("Low Churn Risk — Customer likely to stay.")

    st.markdown("---")

    # Customer segmentation
    st.subheader("Customer Segment Analyzer (K-Means Clustering)")
    seg_features = ["EngagementScore", "CustomerSatisfaction", "MonthlyCharges"]
    seg_df = df[seg_features].copy()
    kmeans = KMeans(n_clusters=3, random_state=42)
    seg_df["Cluster"] = kmeans.fit_predict(seg_df)
    seg_df["Segment"] = seg_df["Cluster"].map({0: "Low Value", 1: "Mid Value", 2: "High Value"})
    fig_seg = px.scatter_3d(seg_df, x="EngagementScore", y="CustomerSatisfaction", z="MonthlyCharges",
                            color="Segment", title="Customer Segmentation (3D Cluster View)")
    st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · Scikit-learn · Pandas")

if __name__ == "__main__":
    main()
