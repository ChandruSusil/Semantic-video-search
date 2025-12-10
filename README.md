# Semantic Video Search Engine 🎥 🔍

A "YouTube Spotter" tool that allows you to search inside videos using natural language descriptions. This project demonstrates Multimodal AI using **OpenAI CLIP**, **Faiss**, and **OpenCV**.

## Features
- **Semantic Search**: Find actions (e.g., "chef chopping onions") without manual tags.
- **Multimodal AI**: Bridges the gap between text and video using the CLIP model.
- **Efficient Indexing**: Uses Faiss for high-speed vector similarity search.
- **Streamlit UI**: Simple and modern web interface.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ChandruSusil/Semantic-video-search.git
    cd Semantic-video-search
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install PyTorch, Transformers, and other ML libraries.*

## Usage

1.  Run the Streamlit app:
    ```bash
    streamlit run semantic_video_search/app.py
    ```
2.  Paste a YouTube URL or upload a video file.
3.  Click **"Extract & Index Frames"**.
4.  Type your query (e.g., "A red car drifting") and adjust the confidence threshold.

## Tech Stack
- **Python 3.x**
- **OpenAI CLIP** (via Hugging Face Transformers)
- **Faiss** (Vector Database)
- **OpenCV** (Video Processing)
- **Streamlit** (Frontend)

## License
MIT License
