import io

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib.figure import Figure
from pedalboard import Chorus
from pedalboard import Compressor
from pedalboard import Convolution
from pedalboard import Distortion
from pedalboard import Gain
from pedalboard import HighpassFilter
from pedalboard import LadderFilter
from pedalboard import Limiter
from pedalboard import LowpassFilter
from pedalboard import Pedalboard
from pedalboard import Phaser
from pedalboard import Reverb
from scipy.io import wavfile


def _normalize_16_bit(audio_data: np.ndarray) -> np.ndarray:
    """Not sure why, but with soundfile, input data should be normalized for 16 bitbefore Streamlit can play it. Not needed after Pedalboard render.
    * https://stackoverflow.com/questions/57925304/how-to-normalize-a-raw-audio-file-with-python
    * https://github.com/jkanner/streamlit-audio/blob/main/helper.py
    """
    return np.int16(audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9)


def _audio_to_virtualfile(
    audio_data: np.ndarray, normalize_16: bool = False
) -> io.BytesIO:
    """Write audio to virtual file so Streamlit audio player can render"""
    if normalize_16:
        audio_data = _normalize_16_bit(audio_data)
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, 44100, audio_data)
    return virtualfile


def _plot_waveform(audio_data: np.ndarray, sample_rate: int) -> Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
    return fig


def main():
    ### -------------------------- Sidebar checkboxes -------------------------- ###
    with st.sidebar:
        enable_convolution = st.checkbox("Enable Convolution", False)
        enable_compressor = st.checkbox("Enable Compressor", False)
        enable_chorus = st.checkbox("Enable Chorus", False)
        enable_distortion = st.checkbox("Enable Distortion", False)
        enable_gain = st.checkbox("Enable Gain", True)
        enable_hp = st.checkbox("Enable Highpass Filter", False)
        enable_ld = st.checkbox("Enable Ladder Filter", False)
        enable_limiter = st.checkbox("Enable Limiter", False)
        enable_lp = st.checkbox("Enable Lowpass Filter", False)
        enable_phaser = st.checkbox("Enable Phaser", False)
        enable_reverb = st.checkbox("Enable Reverb", False)

    ### -------------------------- Import file -------------------------- ###
    st.markdown(
        """
Testing [Spotify Pedalboard](https://github.com/spotify/pedalboard).
    """
    )
    st.subheader(":file_folder: Import music file")
    st.caption("...or use the sample example")
    c1, c2 = st.columns((2, 1))
    uploaded_file = c1.file_uploader("Upload wav file", type=["wav"])
    if uploaded_file is None:
        audio, sample_rate = librosa.load(librosa.example("vibeace", hq=True), sr=44100)
    else:
        audio, sample_rate = librosa.load(uploaded_file, sr=44100)

    st.pyplot(_plot_waveform(audio, sample_rate))

    c2.caption("Currently loaded audio file")
    c2.audio(_audio_to_virtualfile(audio))

    st.write("---")

    ### -------------------------- Configure effects -------------------------- ###
    st.subheader(":hammer_and_wrench: Configure effects")
    board = Pedalboard([], sample_rate=sample_rate)

    if enable_convolution:
        with st.expander("Convolution"):
            st.warning("Not yet implemented")
            pass

    if enable_compressor:
        with st.expander("Compressor"):
            selected_compressor_threshold_db = st.slider(
                "Threshold in dB", -24.0, 24.0, Compressor().threshold_db
            )
            selected_compressor_ratio = st.slider(
                "Ratio", 0.0, 10.0, Compressor().ratio
            )
            selected_compressor_attack_ms = st.slider(
                "Attack in ms", 0.1, 100.0, Compressor().attack_ms
            )
            selected_compressor_release_ms = st.slider(
                "Release in ms", 0.1, 1000.0, Compressor().release_ms
            )
            board.append(
                Compressor(
                    threshold_db=selected_compressor_threshold_db,
                    ratio=selected_compressor_ratio,
                    attack_ms=selected_compressor_attack_ms,
                    release_ms=selected_compressor_release_ms,
                )
            )

    if enable_chorus:
        with st.expander("Chorus"):
            st.warning("Not yet implemented")
            pass

    if enable_distortion:
        with st.expander("Distortion"):
            st.warning("Not yet implemented")
            pass

    if enable_gain:
        with st.expander("Gain"):
            selected_gain_db = st.slider("Gain in dB", -24.0, 24.0, Gain().gain_db)
            board.append(Gain(gain_db=selected_gain_db))

    if enable_hp:
        with st.expander("Highpass Filter"):
            st.warning("Not yet implemented")
            pass

    if enable_ld:
        with st.expander("Ladder Filter"):
            st.warning("Not yet implemented")
            pass

    if enable_limiter:
        with st.expander("Limiter"):
            st.warning("Not yet implemented")
            pass

    if enable_lp:
        with st.expander("Lowpass Filter"):
            st.warning("Not yet implemented")
            pass

    if enable_phaser:
        with st.expander("Phaser"):
            st.warning("Not yet implemented")
            pass

    if enable_reverb:
        with st.expander("Reverb"):
            selected_reverb_room_size = st.slider(
                "Room Size", 0.0, 1.0, Reverb().room_size
            )
            selected_reverb_damping = st.slider("Damping", 0.0, 1.0, Reverb().damping)
            selected_reverb_wet_level = st.slider(
                "Wet Level", 0.0, 1.0, Reverb().wet_level
            )
            selected_reverb_dry_level = st.slider(
                "Dry Level", 0.0, 1.0, Reverb().dry_level
            )
            selected_reverb_width = st.slider("Width", 0.0, 1.0, Reverb().width)
            selected_reverb_freeze_mode = st.slider(
                "Freeze mode", 0.0, 1.0, Reverb().freeze_mode
            )
            board.append(
                Reverb(
                    room_size=selected_reverb_room_size,
                    damping=selected_reverb_damping,
                    wet_level=selected_reverb_wet_level,
                    dry_level=selected_reverb_dry_level,
                    width=selected_reverb_width,
                    freeze_mode=selected_reverb_freeze_mode,
                )
            )

    with st.sidebar:
        st.subheader("Debug")
        st.write(board)

    st.write("---")

    ### -------------------------- Pedalboard render -------------------------- ###
    st.subheader(":headphones: Pedalboard output")
    effected = board(audio)
    st.pyplot(_plot_waveform(effected, sample_rate))
    st.audio(_audio_to_virtualfile(effected))
    st.download_button(
        "Dowload effected",
        _audio_to_virtualfile(effected),
        file_name="output.wav",
        mime="audio/wave",
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Pedalboard demo", page_icon="musical_note")
    st.title("Spotify Pedalboard")
    st.sidebar.subheader("Configuration")
    main()
