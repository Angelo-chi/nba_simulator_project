import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df["Rk"] = pd.to_numeric(df["Rk"], errors='coerce')
    df = df[df["Rk"].notnull()]
    df = df[~df["Player"].str.contains("League Average", na=False)]
    for stat in ["PTS", "AST", "TRB"]:
        df[stat] = pd.to_numeric(df[stat], errors='coerce')
    return df

def get_player_data(df, player_name):
    player_df = df[df["Player"].str.lower() == player_name.lower()]
    if player_df.empty:
        return None
    player_df = player_df.sort_values("Season")
    player_df["Year"] = player_df["Season"].str[:4].astype(int)
    player_df["Experience"] = player_df["Year"] - player_df["Year"].min() + 1
    return player_df

def train_stat_model(df, stat, position, team):
    team_pos_df = df[(df["Team"] == team) & (df["Pos"].str.contains(position, na=False))].copy()
    team_pos_df = team_pos_df.dropna(subset=["Season", stat])
    team_pos_df["Year"] = team_pos_df["Season"].str[:4].astype(int)
    team_pos_df["Experience"] = team_pos_df.groupby("Player")["Year"].rank(method="dense").astype(int)

    X = team_pos_df[["Year", "Experience"]]
    y = team_pos_df[stat]

    if len(X) < 3:
        return None

    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model

def predict_stat(model, years, experience_list):
    X_pred = pd.DataFrame({
        "Year": years,
        "Experience": experience_list
    })
    return model.predict(X_pred) if model else [None] * len(years)

def main():
    st.title("🏀 NBA 新秀養成模擬器")
    st.markdown("模擬一位球員如果被不同球隊選中，5 年內的數據會怎麼變化")

    data_file = "nba_per_game_2004_2023.csv"
    df = load_and_clean_data(data_file)

    player_name = st.text_input("輸入球員英文名字：", "")
    input_team = st.text_input("模擬球隊縮寫（例：LAL, BOS）", "").upper()

    if st.button("開始模擬") and player_name and input_team:
        with st.spinner("🔄 模擬模型啟動中，請稍候 10～30 秒..."):
            player_data = get_player_data(df, player_name)
            if player_data is None:
                st.error("❌ 找不到這位球員，請確認拼寫正確。")
                return

            orig_team = player_data["Team"].mode()[0]
            player_position = player_data["Pos"].mode()[0]
            orig_team_data = player_data[player_data["Team"] == orig_team]

            st.subheader(f"🎯 {player_name} 在 {orig_team} 的實際數據")
            st.write(orig_team_data[["Season", "PTS", "AST", "TRB"]])

            season_years = [int(s.split('-')[0]) for s in orig_team_data["Season"]]
            experience_list = list(range(1, len(season_years) + 1))

            sim_stats = orig_team_data[["Season"]].copy()
            for stat in ["PTS", "AST", "TRB"]:
                model = train_stat_model(df, stat, player_position, input_team)
                sim_stats[stat] = predict_stat(model, season_years, experience_list)

            st.subheader(f"🔮 模擬：{player_name} 如果加入 {input_team} 的數據預測")
            st.write(sim_stats)

            stats = ["PTS", "AST", "TRB"]
            for stat in stats:
                fig, ax = plt.subplots()
                ax.plot(season_years, orig_team_data[stat], marker='o', label=f"{stat} 實際")
                ax.plot(season_years, sim_stats[stat], linestyle='--', marker='^', label=f"{stat} 模擬（{input_team}）")
                ax.set_title(f"{player_name} - {stat} 成長比較")
                ax.set_xlabel("年份")
                ax.set_ylabel(stat)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
