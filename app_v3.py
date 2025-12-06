import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os


# 데이터 로드 함수
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    if "YEAR" in df.columns:
        # 소수점 제거 후 정수로 변환
        df["YEAR_INT"] = df["YEAR"].astype(int)
        df["DATE"] = pd.to_datetime(df["YEAR_INT"].astype(str), format="%Y")
        df.set_index("DATE", inplace=True)
    return df


# 시각화 함수 (인자 추가됨: hist_bins, trend_degree, point_size, point_alpha)
def plot_advanced_sunspot_visualizations(
    df,
    sunactivity_col="SUNACTIVITY",
    hist_bins=30,
    trend_degree=1,
    point_size=10,
    point_alpha=0.5,
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)

    # (a) 전체 시계열 라인 차트
    axs[0, 0].plot(df.index, df[sunactivity_col], color="blue", linewidth=1)
    axs[0, 0].set_title("Sunspot Activity Over Time")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_ylabel("Sunspot Count")
    axs[0, 0].grid(True)

    # (b) 분포: 히스토그램 + 커널 밀도
    data = df[sunactivity_col].dropna().values
    if len(data) > 0:  # 데이터가 있는지 확인
        xs = np.linspace(data.min(), data.max(), 200)
        density = gaussian_kde(data)

        # 히스토그램에 hist_bins 적용
        axs[0, 1].hist(
            data,
            bins=hist_bins,
            density=True,
            alpha=0.6,
            color="gray",
            label="Histogram",
        )
        axs[0, 1].plot(xs, density(xs), color="red", linewidth=2, label="Density")

    axs[0, 1].set_title("Distribution of Sunspot Activity")
    axs[0, 1].set_xlabel("Sunspot Count")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # (c) 상자 그림: 1900년~2000년
    try:
        df_20th = df.loc["1900":"2000"]
        if not df_20th.empty:
            axs[1, 0].boxplot(df_20th[sunactivity_col], vert=False)
    except:
        pass

    axs[1, 0].set_title("Boxplot of Sunspot Activity (1900-2000)")
    axs[1, 0].set_xlabel("Sunspot Count")

    # (d) 산점도 + 회귀선
    years = df["YEAR"].values
    sun_activity = df[sunactivity_col].values

    # NaN 값 제거
    mask = ~np.isnan(sun_activity)
    years_clean = years[mask]
    sun_activity_clean = sun_activity[mask]

    if len(years_clean) > 1:  # 회귀선을 그리기 위해 최소 2개 이상의 데이터 필요
        # 산점도에 point_size, point_alpha 적용
        axs[1, 1].scatter(
            years_clean,
            sun_activity_clean,
            s=point_size,
            alpha=point_alpha,
            label="Data Points",
        )

        # 추세선 차수에 trend_degree 적용
        coef = np.polyfit(years_clean, sun_activity_clean, trend_degree)
        trend = np.poly1d(coef)

        # 추세선을 그리기 위한 x 값 생성 (부드러운 곡선을 위해)
        x_trend = np.linspace(years_clean.min(), years_clean.max(), 100)
        axs[1, 1].plot(
            x_trend, trend(x_trend), color="red", linewidth=2, label="Trend Line"
        )

    axs[1, 1].set_title("Trend of Sunspot Activity")
    axs[1, 1].set_xlabel("Year")
    axs[1, 1].set_ylabel("Sunspot Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# 메인 앱
st.title("🌞 태양흑점 데이터 분석 대시보드 (Interactive) 🌞")
st.markdown(
    """
    이 대시보드는 태양흑점 데이터를 다양한 시각화 방법으로 보여주며, 
    사이드바를 통해 시각화 옵션을 조절할 수 있습니다.
    """
)

try:
    # 데이터 로드
    file_path = "sunspots.csv"
    if not os.path.exists(file_path):
        file_path = "../sunspots.csv"
    if not os.path.exists(file_path):
        file_path = "data/sunspots.csv"

    df = load_data(file_path)

    # 사이드바에 파라미터 조절 슬라이더 추가
    st.sidebar.header("시각화 설정")

    # 연도 범위 선택
    min_year = int(df["YEAR"].min())
    max_year = int(df["YEAR"].max())
    year_range = st.sidebar.slider(
        "연도 범위 선택",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    # 히스토그램 빈(bin) 수 조절
    hist_bins = st.sidebar.slider(
        "히스토그램 구간 수", min_value=5, max_value=100, value=30
    )

    # 추세선 차수 조절
    trend_degree = st.sidebar.slider("추세선 차수", min_value=1, max_value=5, value=1)

    # 산점도 점 크기 조절
    point_size = st.sidebar.slider(
        "산점도 점 크기", min_value=1, max_value=100, value=10
    )

    # 산점도 투명도 조절
    point_alpha = st.sidebar.slider(
        "산점도 투명도", min_value=0.1, max_value=1.0, value=0.5
    )

    # 필터링된 데이터
    filtered_df = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]

    # 시각화
    if not filtered_df.empty:
        st.subheader(f"태양흑점 데이터 종합 시각화 ({year_range[0]} ~ {year_range[1]})")
        fig = plot_advanced_sunspot_visualizations(
            filtered_df,
            hist_bins=hist_bins,
            trend_degree=trend_degree,
            point_size=point_size,
            point_alpha=point_alpha,
        )
        st.pyplot(fig)
    else:
        st.warning("선택한 기간에 데이터가 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
    st.info("데이터 파일이 올바른 위치에 있는지 확인해주세요.")
