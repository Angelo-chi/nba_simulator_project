import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import random
from xgboost import XGBRegressor
from time import time
import os

# 設置頁面
st.set_page_config(page_title="NBA 模擬器", layout="centered")

# 封面圖片連結清單
cover_images = [
    "https://fadeawayworld.net/.image/t_share/MTg5MjE3NzgyNjAwNTAzNDg2/stephcoldest.webp",
    "https://larrybrownsports.com/wp-content/uploads/2024/04/lebron-james-usa.jpg",
    "https://i.ytimg.com/vi/-H9HrmKL2eQ/maxresdefault.jpg",
    "https://fadeawayworld.net/.image/t_share/MTkzMzk3NDg0MDkzNzAzNzQy/tatum-coldest.webp"
]

@st.cache_data
def load_and_clean_data(url="https://drive.google.com/uc?export=download&id=1zSkbqzuloQHY2P-75c98m_cD5TBzqK5I", filepath="nba_per_game_2004_2023.csv"):
    # 從雲端下載 CSV（僅第一次執行）
    if not os.path.exists(filepath):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 檢查下載是否成功
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            st.error(f"❌ 無法下載資料檔案：{e}")
            return None

    # 分塊讀取 CSV，減少記憶體使用
    chunks = pd.read_csv(
        filepath,
        usecols=["Player", "Season", "Team", "Pos", "Rk", "PTS", "AST", "TRB"],
        chunksize=10000
    )
    df = pd.concat([chunk for chunk in chunks], ignore_index=True)
    
    # 清理資料
    df["Rk"] = pd.to_numeric(df["Rk"], errors='coerce')
    df = df[df["Rk"].notnull()]
    df = df[~df["Player"].str.contains("League Average", na=False)]
    for stat in ["PTS", "AST", "TRB"]:
        df[stat] = pd.to_numeric(df[stat], errors='coerce')
    return df

def get_player_data(df, player_name):
    if df is None:
        return None
    player_df = df[df["Player"].str.lower() == player_name.lower()]
    if player_df.empty:
        return None
    player_df = player_df.sort_values("Season")
    player_df["Year"] = player_df["Season"].str[:4].astype(int)
    player_df["Experience"] = player_df["Year"] - player_df["Year"].min() + 1
    return player_df

@st.cache_resource
def train_stat_model(df, stat, position, team):
    if df is None:
        return None
    team_pos_df = df[(df["Team"] == team) & (df["Pos"].str.contains(position, na=False))].copy()
    team_pos_df = team_pos_df.dropna(subset=["Season", stat])
    team_pos_df["Year"] = team_pos_df["Season"].str[:4].astype(int)
    team_pos_df["Experience"] = team_pos_df.groupby("Player")["Year"].rank(method="dense").astype(int)

    if len(team_pos_df) < 3:
        return None

    X = team_pos_df[["Year", "Experience"]]
    y = team_pos_df[stat]
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model

def predict_stat(model, years, experience_list):
    X_pred = pd.DataFrame({
        "Year": years,
        "Experience": experience_list
    })
    return model.predict(X_pred) if model else [None] * len(years)

# 主程式
st.title("🏀 NBA 新秀模擬器")
st.image(random.choice(cover_images), caption="📸 The Coldest Moment", use_column_width=True)
st.markdown("""
### 👋 歡迎使用 NBA 新秀模擬器！
> 本模擬器根據 2004-2023 的 NBA 球員數據，模擬球員若在不同球隊成長的表現（PTS、AST、TRB）。
請輸入球員英文名字與模擬球隊縮寫（例如：**LAL**, **BOS**, **GSW**）。
""")

# 熱門球員按鈕區
st.markdown("#### 🔥 熱門球員快速選擇")
popular_players = ["Jayson Tatum", "Luka Doncic", "Victor Wembanyama", "Zion Williamson", "Stephen Curry", "LeBron James", "Paul George"]
cols = st.columns(len(popular_players))
selected_name = ""
for i, player in enumerate(popular_players):
    if cols[i].button(player):
        selected_name = player

# 使用者輸入
player_name = st.text_input("球員英文名字", value=selected_name if selected_name else "")
input_team = st.text_input("模擬球隊縮寫（例如：LAL, BOS）", value="LAL").upper()

# 限制模擬頻率
if "last_simulated" not in st.session_state:
    st.session_state.last_simulated = 0

# 模擬邏輯
if st.button("🔄 開始模擬") and player_name and input_team:
    if time() - st.session_state.last_simulated < 10:
        st.warning("⚠️ 請等待 10 秒後再試！")
    else:
        st.session_state.last_simulated = time()
        with st.spinner("🔄 模擬模型啟動中，請稍候 10～30 秒..."):
            df = load_and_clean_data()
            if df is None:
                st.error("❌ 無法載入資料，請稍後再試。")
            elif input_team not in df["Team"].unique():
                st.error(f"❌ 無效球隊縮寫，請輸入有效縮寫，例如：{', '.join(df['Team'].unique()[:5])}")
            else:
                player_data = get_player_data(df, player_name)
                if player_data is None:
                    st.error("❌ 找不到這位球員，請確認拼寫正確。")
                else:
                    orig_team = player_data["Team"].mode()[0]
                    player_position = player_data["Pos"].mode()[0]
                    orig_team_data = player_data[player_data["Team"] == orig_team]

                    st.subheader(f"🎯 {player_name} 在 {orig_team} 的實際數據")
                    st.dataframe(orig_team_data[["Season", "PTS", "AST", "TRB"]])

                    season_years = [int(s.split('-')[0]) for s in orig_team_data["Season"]]
                    experience_list = list(range(1, len(season_years) + 1))

                    sim_stats = orig_team_data[["Season"]].copy()
                    for stat in ["PTS", "AST", "TRB"]:
                        model = train_stat_model(df, stat, player_position, input_team)
                        if model is None:
                            st.warning(f"⚠️ {input_team} 的 {player_position} 位置資料不足，無法模擬 {stat}。")
                            sim_stats[stat] = [None] * len(season_years)
                        else:
                            sim_stats[stat] = predict_stat(model, season_years, experience_list)

                    st.subheader(f"🔮 模擬：{player_name} 如果加入 {input_team} 的數據預測")
                    st.dataframe(sim_stats)

                    # 使用 Plotly 繪製圖表
                    stats = ["PTS", "AST", "TRB"]
                    for stat in stats:
                        if sim_stats[stat].isnull().all():
                            continue
                        fig = px.line(
                            x=season_years,
                            y=[orig_team_data[stat], sim_stats[stat]],
                            labels={"x": "年份", "value": stat, "variable": "數據類型"},
                            title=f"{player_name} - {stat} 成長比較"
                        )
                        fig.update_traces(mode="lines+markers", selector=dict(name="0"), name=f"{stat} 實際")
                        fig.update_traces(mode="lines+markers", line_dash="dash", selector=dict(name="1"), name=f"{stat} 模擬（{input_team}）")
                        st.plotly_chart(fig, use_container_width=True)