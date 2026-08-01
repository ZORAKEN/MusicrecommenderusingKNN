import pickle

import spotipy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="🎵 Find your beat",
    page_icon="🎵",
    layout="wide",
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.main{
    background:#0E1117;
}

.hero{
    text-align:center;
    padding:30px;
    border-radius:15px;
    background:linear-gradient(90deg,#6C63FF,#1DB954);
    color:white;
    margin-bottom:25px;
}

.song-card{
    background:#161B22;
    border:1px solid #30363d;
    border-radius:18px;
    overflow:hidden;
    padding:10px;
    transition:.3s;
}

.song-card:hover{
    transform:translateY(-8px);
    box-shadow:0 10px 30px rgba(29,185,84,.45);
}

.song-name{
    font-size:18px;
    font-weight:bold;
    color:white;
    margin-top:8px;
}

.artist-name{
    color:#BBBBBB;
    font-size:14px;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:12px;
    background:#1DB954;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#18a64c;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🎵 Find your beat</h1>
    <h4>Get your taste.</h4>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SPOTIFY CONFIGURATION
# ==========================================================

CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_music():
    with open("df.pkl", "rb") as file:
        return pickle.load(file)

music = load_music()

# ==========================================================
# CREATE TF-IDF MATRIX
# ==========================================================

tfidf = TfidfVectorizer(
    max_features=30000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(music["text"])

# ==========================================================
# GET ALBUM COVER
# ==========================================================

def get_album_cover(song, artist):
    """Fetch album artwork from Spotify."""

    try:
        query = f"track:{song} artist:{artist}"

        result = sp.search(
            q=query,
            type="track",
            limit=1
        )

        tracks = result["tracks"]["items"]

        if tracks:
            return tracks[0]["album"]["images"][0]["url"]

    except Exception:
        pass

    return "https://i.postimg.cc/0QNxYz4V/social.png"

# ==========================================================
# RECOMMEND SONGS
# ==========================================================

def recommend(song_name):
    """Return top 5 similar songs."""

    match = music[music["song"] == song_name]

    if match.empty:
        return [], [], []

    index = match.index[0]

    similarity = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    song_indices = similarity.argsort()[::-1][1:6]

    names = []
    artists = []
    posters = []

    for i in song_indices:
        song = music.iloc[i]["song"]
        artist = music.iloc[i]["artist"]

        names.append(song)
        artists.append(artist)
        posters.append(get_album_cover(song, artist))

    return names, posters, artists

# ==========================================================
# USER INPUT
# ==========================================================

selected_song = st.selectbox(
    "🎧 Search for a Song",
    music["song"].values,
    index=None,
    placeholder="Start typing a song..."
)

# ==========================================================
# RECOMMEND BUTTON
# ==========================================================

if st.button("🎵 Recommend Songs"):

    if selected_song is None:
        st.warning("Please select a song.")

    else:
        with st.spinner("Finding similar songs... 🎶"):

            names, posters, artists = recommend(selected_song)

        if names:

            columns = st.columns(5)

            for col, name, poster, artist in zip(
                columns,
                names,
                posters,
                artists,
            ):
                with col:

                    st.markdown(
                        f"""
                        <div class="song-card">
                            <img src="{poster}" width="100%">
                            <div class="song-name">{name}</div>
                            <div class="artist-name">🎤 {artist}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        else:
            st.error("Song not found.")

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3659/3659898.png",
        width=120
    )

    st.header("About")

    st.write("""
This recommendation system uses:

- TF-IDF Vectorization
- Cosine Similarity
- Spotify API

to recommend songs with similar lyrics and musical styles.
""")

    st.divider()

    st.info("🎵 Select a song and click **Recommend Songs**.")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <hr>
    <center>
    Made at 2 am while listening to rain   
    </center>
    """,
    unsafe_allow_html=True,
)