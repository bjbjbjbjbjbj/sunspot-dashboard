import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# 페이지 설정
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")


# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


# 파일 경로 확인 (sunspots_for_prophet.csv)
file_path = "sunspots_for_prophet.csv"
if not os.path.exists(file_path):
    file_path = "../sunspots_for_prophet.csv"
if not os.path.exists(file_path):
    file_path = "data/sunspots_for_prophet.csv"

# 만약 파일이 없으면 원본 데이터로부터 생성 (배포 환경 대비)
if not os.path.exists(file_path):
    original_path = "sunspots.csv"
    if not os.path.exists(original_path):
        original_path = "../sunspots.csv"
    if not os.path.exists(original_path):
        original_path = "data/sunspots.csv"

    if os.path.exists(original_path):
        df_raw = pd.read_csv(original_path)
        df_raw["YEAR"] = df_raw["YEAR"].astype(int)
        df_raw["ds"] = pd.to_datetime(df_raw["YEAR"], format="%Y")
        df = df_raw.rename(columns={"SUNACTIVITY": "y"})
        df = df[(df["ds"] >= "1900-01-01") & (df["ds"] <= "2008-01-01")]
    else:
        st.error("데이터 파일을 찾을 수 없습니다.")
        st.stop()
else:
    df = load_data(file_path)

st.subheader("📄 데이터 미리보기")
st.write(df.head())

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
model = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.05)
model.add_seasonality(name="sunspot_cycle", period=11, fourier_order=5)
model.fit(df)

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
future = model.make_future_dataframe(periods=30, freq="Y")
forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
fig1 = model.plot(forecast)
plt.title("Prophet Forecast Plot")
plt.xlabel("Year")
plt.ylabel("Sun Activity")
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
plt.suptitle("Forecast Components", fontsize=16)
st.pyplot(fig2)

# ----------------------------------
# [5] 커스텀 시각화: 실제값 vs 예측값 + 신뢰구간
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")

fig3, ax = plt.subplots(figsize=(14, 6))

# 실제 데이터 (파란색 실선 + 마커) - df 사용 (1900년 이후)
ax.plot(
    df["ds"],
    df["y"],
    label="Actual",
    color="blue",
    marker="o",
    linestyle="-",
    markersize=5,
)
# 예측 데이터 (빨간색 점선)
ax.plot(
    forecast["ds"],
    forecast["yhat"],
    label="Predicted",
    color="red",
    linestyle="--",
    linewidth=2,
)
# 신뢰구간 (빨간색 영역)
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="red",
    alpha=0.1,
    label="Prediction Interval",
)

ax.set_title("Sunspots: Actual vs. Predicted with Prediction Intervals")
ax.set_xlabel("Year")
ax.set_ylabel("Sun Activity")
ax.legend(loc="upper right")
ax.grid(True)
st.pyplot(fig3)

# ----------------------------------
# [6] 잔차 분석 시각화
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")

# 잔차 계산
merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="inner")
merged["residual"] = merged["y"] - merged["yhat"]

# 잔차 시각화
fig4, ax2 = plt.subplots(figsize=(14, 4))
# 잔차 시각화 (자주색 실선 + 마커)
ax2.plot(
    merged["ds"],
    merged["residual"],
    label="Residual",
    color="purple",
    marker="o",
    linestyle="-",
    markersize=5,
)
ax2.axhline(0, color="black", linestyle="--", linewidth=1.5)

ax2.set_title("Residual Analysis (Actual - Predicted)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True)
st.pyplot(fig4)

# ----------------------------------
# [7] 잔차 통계 요약 출력
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")
st.write(merged["residual"].describe())
