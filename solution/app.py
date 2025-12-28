import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv 
from openai import AzureOpenAI
import os 
from huggingface_hub import hf_hub_download 

load_dotenv()

@st.cache_data
def load_data_and_embeddings():
    data = pd.read_csv("dataset/dataset_cleaned.csv")
    data = data.reset_index(drop=True)

    emb_path = hf_hub_download(
        repo_id = "Tungtom2004/dataset",
        filename = "embeddings_hybrid.npy",
        repo_type="dataset",
    )
    embeddings = np.load(emb_path, mmap_mode="r").astype("float16")
    return data, embeddings


client = AzureOpenAI(
    api_key = st.secrets["AZURE_OPENAI_KEY"],
    api_version = st.secrets["AZURE_OPENAI_VERSION"],
    azure_endpoint= st.secrets["AZURE_OPENAI_ENDPOINT"],
    azure_deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
)

def explain_recommendation(base_track, rec_tracks):
    songs_text = ""
    for _, row in rec_tracks.iterrows():
        songs_text += f"- {row.track_name} by {row.artists_str} (genre: {row.track_genre})\n"

    prompt = f"""
You are a music recommendation assistant.

User listened to:
"{base_track.track_name}" by {base_track.artists_str}
Genre: {base_track.track_genre}
Energy: {base_track.energy}
Tempo: {base_track.tempo}

The system recommended these songs:
{songs_text}

REQUIREMENTS:
- Explain the REASON for EACH song in the list (DO NOT omit any)
- 1-2 short sentences per song
- Friendly, easy-to-understand tone, like a conversation with the user
- No technical jargon
- No mechanical repetition of information
- Write in VIETNAMESE
- Present in bulleted list format (•)

Example of writing:

• Because of its similar lively melody, this song helps maintain the same feeling when listening.

• With the same style and rhythm, it's very suitable for listening again.
"""

    response = client.chat.completions.create(
        model=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": "You explain music recommendations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content



data, embeddings_hybrid = load_data_and_embeddings()

def search_tracks(query, limit=20):
    if not query:
        return pd.DataFrame()

    mask = data["track_name"].str.contains(query, case=False, na=False)
    results = data[mask].copy()
    results = results.drop_duplicates(subset=["track_name", "artists_str"]).head(limit)
    return results


def recommend_from_position(pos, top_k=10, embeddings=None):
    if embeddings is None:
        embeddings = embeddings_hybrid

    query_vec = embeddings[pos].reshape(1, -1)
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    sim_scores[pos] = -1

    sorted_idx = np.argsort(sim_scores)[::-1]

    candidates = data.iloc[sorted_idx][[
        "track_name", "artists_str", "track_genre"
    ]].copy()

    candidates["similarity"] = sim_scores[sorted_idx]

    candidates = candidates.drop_duplicates(subset=["track_name", "artists_str"])

    return candidates.head(top_k).reset_index(drop=True)


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

body {
    background-color: #121212 !important;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.sidebar .sidebar-content {
    background-color: #000 !important;
    color: white;
    padding: 20px;
}

.sidebar .sidebar-content h2, .sidebar .sidebar-content p {
    color: white !important;
}

.stButton>button {
    background-color: #1DB954 !important;
    color: white !important;
    border-radius: 25px;
    padding: 10px 25px;
    border: none;
    font-weight: 600;
    transition: 0.2s;
}

.stButton>button:hover {
    background-color: #1ed760 !important;
    transform: scale(1.05);
}

.big-title {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(90deg,#1DB954,#1ed760);
    -webkit-background-clip: text;
    color: transparent;
}

.song-card {
    background-color: #181818;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 14px;
    border: 1px solid #2a2a2a;
    transition: 0.15s ease-in-out;
}

.song-card:hover {
    background-color: #232323;
    border-color: #1DB954;
    transform: scale(1.02);
}

.song-title {
    font-size: 18px;
    font-weight: 700;
    color: #fff;
}

.song-artist {
    font-size: 14px;
    color: #b3b3b3;
}

.song-genre {
    font-size: 13px;
    color: #1DB954;
    margin-bottom: 8px;
}

.sim-value {
    font-size: 13px;
    color: #ffd369;
}

.album-cover {
    width: 100%;
    border-radius: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["🎵 Music Recommender", "📊 Visualization"])

with tab1:

    st.markdown("<h1 class='big-title'>🎵 Music Recommendation System</h1>", unsafe_allow_html=True)
    st.write("Tìm bài hát & nhận gợi ý theo **Real-time + Context-aware + User History**.")


    if "history" not in st.session_state:
        st.session_state.history = []  

    col_left, col_right = st.columns([1.2, 2])

    with col_left:
        st.subheader("🔍 Tìm bài hát")

        query = st.text_input(
            "Nhập tên bài hát",
            key="search",
            placeholder="Ví dụ: Shape of You"
        )

        results = search_tracks(query)

        selected_index = None

        if not results.empty:
            labels = [f"{row.track_name} — {row.artists_str}" for _, row in results.iterrows()]
            selection = st.selectbox("Chọn bài hát:", labels)

            if selection:
                selected_index = results.iloc[labels.index(selection)].name


        st.subheader("🎚 Context Mode")

        context = st.selectbox(
            "Chọn chế độ nghe:",
            ["Default", "Chill", "Workout", "Sleep", "Energy Boost"]
        )

        st.write("📌 Context sẽ ảnh hưởng đến gợi ý dựa trên audio features.")

        top_k = st.slider("Số lượng gợi ý (K):", 5, 30, 10)


        st.subheader("🕒 Lịch sử bài hát đã xem gợi ý")

        if len(st.session_state.history) == 0:
            st.info("Chưa có lịch sử.")
        else:
            for item in reversed(st.session_state.history[-10:]):
                st.write("•", item)


    with col_right:
        st.subheader("🎧 Gợi ý theo thời gian thực")

        if selected_index is not None:

            # Lưu lịch sử (nếu chưa có)
            chosen_track = f"{data.loc[selected_index, 'track_name']} — {data.loc[selected_index, 'artists_str']}"
            if chosen_track not in st.session_state.history:
                st.session_state.history.append(chosen_track)

            # Hiển thị bài hát gốc
            track = data.loc[selected_index]

            st.markdown(f"""
            <div class='song-card'>
                <div class='song-title'>{track.track_name}</div>
                <div class='song-artist'>{track.artists_str}</div>
                <div class='song-genre'>Genre: {track.track_genre}</div>
            </div>
            """, unsafe_allow_html=True)

            # ============================
            # CONTEXT FILTERING FUNCTION
            # ============================

            def apply_context_filter(df, ctx):
                if ctx == "Default":
                    return df

                if ctx == "Chill":
                    return df[
                        (df["energy"] < 0.5) &
                        (df["acousticness"] > 0.4)
                    ]

                if ctx == "Workout":
                    return df[
                        (df["energy"] > 0.7) &
                        (df["tempo"] > 120)
                    ]

                if ctx == "Sleep":
                    return df[
                        (df["liveness"] < 0.25) &
                        (df["speechiness"] < 0.4)
                    ]

                if ctx == "Energy Boost":
                    return df[
                        (df["energy"] > 0.8) |
                        (df["danceability"] > 0.7)
                    ]

                return df

            # =====================================================
            # CREATE RECOMMENDATION BASED ON CONTEXT + REAL TIME
            # =====================================================

            # Bản gốc của rec
            base_recs = recommend_from_position(selected_index, top_k=200)

            # Lọc theo context
            filtered_recs = apply_context_filter(
                base_recs.merge(data, on=["track_name", "artists_str", "track_genre"]),
                context
            )
            filtered_recs = filtered_recs.drop_duplicates(subset = ["track_name","artists_str"])

            # Lấy lại K bài sau lọc
            final_recs = filtered_recs.head(top_k)

            # ============================
            # SHOW RESULTS
            # ============================

            if len(final_recs) == 0:
                st.warning("Không tìm được bài phù hợp với context hiện tại.")
            else:
                st.markdown("### 🔥 Gợi ý theo bài của người dùng tìm kiếm:")

                for _, row in final_recs.iterrows():
                    st.markdown(f"""
                    <div class='song-card'>
                        <div class='song-title'>{row.track_name}</div>
                        <div class='song-artist'>{row.artists_str}</div>
                        <div class='song-genre'>Genre: {row.track_genre}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 🤖AI Explanation Context")
                if st.button("Vì sao hệ thống gợi ý các bài này?"):
                    with st.spinner("..."):
                        explanation = explain_recommendation(
                            base_track=track,
                            rec_tracks=final_recs
                        )
                    st.info(explanation)
        else:
            st.info("👉 Nhập bài hát và chọn 1 bài để xem gợi ý theo thời gian thực.")


# ==========================================
# TAB 2 — VISUALIZATION
# ==========================================

# ==========================================
# TAB 2 — VISUALIZATION
# ==========================================

with tab2:

    st.header("📊 Spotify Data Visualization Dashboard")
    st.write("Các biểu đồ phân tích dữ liệu được xây dựng từ dataset Spotify Tracks của bạn.")

    # ---------------------------------------------------
    # 1. Popularity Distribution
    # ---------------------------------------------------

    st.subheader("1️⃣ Popularity Distribution of Spotify Tracks")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data["popularity"], bins=30, kde=True, color="skyblue", ax=ax)

    mean_val = data["popularity"].mean()
    median_val = data["popularity"].median()

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f"Median = {median_val:.2f}")

    ax.set_title("Popularity Distribution of Spotify Tracks", fontsize=14)
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Count")

    ax.text(
        0.70, 0.80,
        f"Total tracks: {len(data)}\n"
        f"Mean popularity: {mean_val:.2f}\n"
        f"Median popularity: {median_val:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
    )

    ax.legend()
    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------------------------------
    # 2. Genre Occurrences (Top 20)
    # ---------------------------------------------------

    st.subheader("2️⃣ Top 20 Most Common Music Genres")

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_counts = data["track_genre"].value_counts().head(20)

    sns.barplot(
        x=genre_counts.values,
        y=genre_counts.index,
        palette="viridis",
        ax=ax
    )

    ax.set_title("Top 20 Most Common Music Genres", fontsize=14)
    ax.set_xlabel("Count")
    ax.set_ylabel("Genres")

    total_genres = data["track_genre"].nunique()

    ax.text(
        0.70, 0.85,
        f"Total genres: {total_genres}\n"
        f"Top genre: {genre_counts.index[0]} ({genre_counts.values[0]} tracks)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
    )

    # add value labels
    for i, v in enumerate(genre_counts.values):
        ax.text(v + 5, i, str(v), color="black", va="center", fontsize=9)

    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------------------------------
    # 3. Top 20 Artists with Most Tracks
    # ---------------------------------------------------

    st.subheader("3️⃣ Top 20 Artists with the Most Tracks")

    fig, ax = plt.subplots(figsize=(12, 6))

    artist_counts = data["artists_str"].value_counts().head(20)

    sns.barplot(
        x=artist_counts.values,
        y=artist_counts.index,
        palette="magma",
        ax=ax
    )

    ax.set_title("Top 20 Artists with the Most Tracks", fontsize=14)
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Artists")

    total_artists = data["artists_str"].nunique()

    ax.text(
        0.70, 0.85,
        f"Total unique artists: {total_artists}\n"
        f"Top artist: {artist_counts.index[0]} ({artist_counts.values[0]} tracks)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
    )

    for i, v in enumerate(artist_counts.values):
        ax.text(v + 5, i, str(v), color="black", va="center", fontsize=9)

    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------------------------------
    # 4. Correlation Heatmap of Audio Features
    # ---------------------------------------------------

    st.subheader("4️⃣ Audio Feature Correlation Heatmap (+ Notes)")

    numeric_features = [
        "danceability", "energy", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "loudness"
    ]

    corr = data[numeric_features].corr()

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [4, 1]}
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=.5,
        cbar=True,
        ax=ax1
    )

    ax1.set_title("Audio Feature Correlation Heatmap", fontsize=14)

    notes_text = (
        "Notes:\n"
        "- Energy ↗ strongly with Loudness\n"
        "- Acousticness ↘ when Energy ↗\n"
        "- Speechiness has weak correlation\n"
        "- Valence moderately correlates with Danceability\n"
    )

    ax2.text(
        0.1, 0.5,
        notes_text,
        fontsize=12,
        va="center",
        wrap=True
    )
    ax2.axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("5️⃣ Popularity Distribution by Genre (Top Genres)")

    top_genres = data["track_genre"].value_counts().head(5).index
    filtered = data[data["track_genre"].isin(top_genres)]

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(
        data=filtered,
        x="track_genre",
        y="popularity",
        ax=ax
    )

    ax.set_title("Popularity Distribution Across Top Genres")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Popularity")

    st.pyplot(fig)
    
    st.markdown("---")

    st.subheader("6️⃣ The relationship between energy and the danceability of tracks")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(
    data=data.sample(3000),
    x="energy",
    y="danceability",
    hue="popularity",
    palette="viridis",
    alpha=0.6,
    ax=ax
)

    ax.set_title("Energy vs Danceability of Tracks")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Danceability")

    st.pyplot(fig)

    st.markdown("---")



