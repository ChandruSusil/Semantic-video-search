import streamlit as st
import os
import yt_dlp
import glob
from processor import FrameExtractor
from embeddings import CLIPEmbedder
from vector_db import VectorIndex
import numpy as np

# Set page config
st.set_page_config(page_title="Semantic Video Search", layout="wide")

st.title("Semantic Video Search Engine 🎥 🔍")
st.markdown("Search inside videos using natural language! Powered by OpenAI CLIP & Faiss.")

# Initialize components (cached to avoid reloading model)
@st.cache_resource
def get_model():
    return CLIPEmbedder()

@st.cache_resource
def get_index():
    return VectorIndex()

embedder = get_model()
index = get_index()

# Utility to download video
def download_video(url, output_path="downloads"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
        'noplaylist': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)
        return video_path, info_dict.get('title', 'Unknown Title')

# Sidebar for Video Input
with st.sidebar:
    st.header("1. Input Video")
    video_source = st.radio("Choose source:", ["YouTube URL", "Local File"])
    
    video_path = None
    
    if video_source == "YouTube URL":
        url = st.text_input("YouTube URL")
        if st.button("Download & Process"):
            if url:
                with st.spinner("Downloading video..."):
                    try:
                        v_path, title = download_video(url)
                        st.success(f"Downloaded: {title}")
                        st.session_state['video_path'] = v_path
                        st.session_state['video_processed'] = False # Reset processing
                    except Exception as e:
                        st.error(f"Error downloading: {e}")
            
    elif video_source == "Local File":
        uploaded_file = st.file_uploader("Upload MP4", type=["mp4"])
        if uploaded_file:
             # Save to temp
            if not os.path.exists("downloads"):
                os.makedirs("downloads")
            v_path = os.path.join("downloads", uploaded_file.name)
            with open(v_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['video_path'] = v_path
            st.session_state['video_processed'] = False

    if 'video_path' in st.session_state:
        st.info(f"Current Video: {os.path.basename(st.session_state['video_path'])}")
        if st.button("Extract & Index Frames"):
            with st.spinner("Extracting frames..."):
                extractor = FrameExtractor(extraction_interval=2) # 2 seconds
                frames_dir = os.path.join("processed_frames", os.path.splitext(os.path.basename(st.session_state['video_path']))[0])
                frames_data = extractor.extract_frames(st.session_state['video_path'], frames_dir)
                st.write(f"Extracted {len(frames_data)} frames.")
            
            with st.spinner("Embedding & Indexing..."):
                # Reset index for demo purposes (simple single video search)
                index = get_index() 
                index.index.reset()
                index.metadata = []
                
                # Batch processing
                batch_size = 32
                for i in range(0, len(frames_data), batch_size):
                    batch = frames_data[i:i+batch_size]
                    batch_paths = [f[0] for f in batch]
                    batch_meta = batch # Store (path, timestamp) as metadata
                    
                    # Compute embeddings (Image one by one for now or batch if supported)
                    # Our CLIPEmbedder is single image, let's update it or just loop here.
                    # Looping is safer for memory unless we optimized batching in Embedder.
                    vectors = []
                    for f_path in batch_paths:
                        vec = embedder.get_image_embedding(f_path)
                        vectors.append(vec)
                    
                    vectors = np.vstack(vectors)
                    index.add_vectors(vectors, batch_meta)
                    
                st.session_state['video_processed'] = True
                st.success("Indexing Complete!")

# Main Area: Search
st.header("2. Search Video")
if 'video_processed' in st.session_state and st.session_state['video_processed']:
    query = st.text_input("Describe a scene (e.g., 'a cat jumping', 'person laughing')")
    threshold = st.slider("Minimum Relevance Score", 0.0, 1.0, 0.2, 0.01)

    if query:
        text_vec = embedder.get_text_embedding(query)
        # Search more initially (k=10) to have candidates to filter
        results = index.search(text_vec, k=10)
        
        # Filter by threshold
        filtered_results = [res for res in results if res['score'] >= threshold]
        
        if not filtered_results:
            st.warning("No matches found above the threshold. Try lowering the threshold or changing the query.")
        else:
            st.subheader(f"Top Matches (> {threshold}):")
            cols = st.columns(5)
            for i, res in enumerate(filtered_results):
                # Only show top 5 filtered ones to avoid UI mess
                if i >= 5: break
                
                meta = res['metadata']
                image_path = meta[0]
                timestamp = meta[1]
                score = res['score']
                
                with cols[i]:
                    st.image(image_path, caption=f"{int(timestamp)}s (Score: {score:.2f})")
                
                # Optional: Jump to playing video (simple HTML5 video with #t=)
                # Note: Local file playback in Streamlit can be tricky if not in static folder.
                # simpler to just show the image and time for this demo.

else:
    st.info("Please query a video first from the sidebar.")
