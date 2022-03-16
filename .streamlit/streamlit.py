import os, sys
import streamlit as st
import torch
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from settings import Settings
from speed_up import AlgAnd, VolumeThresholdAlgorithm, SileroVadAlgorithm
from main import process_one_video_in_computer
from streamlit.server.server import Server

info = Server.get_current()._session_info_by_id
ident = "_" + list(info.keys())[0] # + str(list(info.values())[0].script_run_count)

st.title("Smart Video Accelerator")
st.write("Collab link: https://colab.research.google.com/drive/1bUevuplQxkqzDnQOEh0MGtTOc5pFBWGY#scrollTo=TH2tbX-JOe1u")
st.write("Github link: https://github.com/mishadobrits/SVA4/tree/dev")

try:
    video = st.file_uploader("Upload a videolecture")
except Exception as e:
    st.write(e)
    video = None


directory = "upload_videos"
os.makedirs(directory, exist_ok=True)
if video:
    videoname = os.path.splitext(video.name)[0]
    videopath = os.path.join(directory, videoname + ident + ".mkv")
    with open(videopath, "wb") as f:
        f.write(video.getbuffer())
    del video

    with st.expander("Parameters"):
        st.write("'↗' means the higher the value, the stronger acceleration (smaller result duration)\n "
                 "'↘' means the higher the value, the weaker acceleration (longer result duration)")
        st.write("↗: From 0 to 1. Parts of the video with volume that smaller then that value will be skipped.")
        volume_threshold = st.number_input("Volume Threshold", value=0.015, min_value=0.0, max_value=1.0, step=0.001, format="%f")
        st.write("↗: From 0.15 to 1. Parts of the video were the Silero algorithm returns probability of speech that smaller then that value will be skipped")
        silero_threshold = st.number_input("Silero Threshold", value=0.45, min_value=0.15, max_value=1.0, step=0.001, format="%f")
        st.write("↗: From 0 to infinity. Parts of the video with speach will be extended on extend_loud_parts seconds, so that ends of phrases won't be cutten.")
        extend_loud_parts = st.number_input("Extend loud parts", value=0.25, min_value=0.0)
        st.write("↘: From 0 to infinity. Parts of the video without speech will be cutten to first max_silent_time seconds for better acceleration.")
        max_silent_time = st.number_input("Max silent time", value=1.0, min_value=extend_loud_parts)


    @st.cache
    def load_silero_vad():
        torch.hub.load(repo_or_dir='snakers4/silero-vad',
                       model='silero_vad',
                       force_reload=False,
                       onnx=True)
        os.system("apt-get install mkvtoolnix")
    load_silero_vad()

    speedup_algorithm = AlgAnd(
        VolumeThresholdAlgorithm(volume_threshold, min_quiet_time=extend_loud_parts),
        SileroVadAlgorithm(silero_threshold),
    )
    settings = Settings(max_quiet_time=max_silent_time, quiet_speed=6)
    output_videopath = os.path.join(directory, "SVA-" + videoname + ident + ".mkv")

    if st.button("Accelerate!"):
        process_one_video_in_computer(videopath, speedup_algorithm, settings, output_videopath)

        def delete_files():
            p1, p2 = os.path.abspath(videopath), os.path.abspath(output_videopath)
            if os.path.exists(p1):
                os.remove(p1)
            if os.path.exists(p2):
                os.remove(p2)

        with open(output_videopath, 'rb') as f:
            st.download_button(
                'Download video', f.read(), "SVA-" + videoname + ".mkv",
                on_click=delete_files
            )
        delete_files()



