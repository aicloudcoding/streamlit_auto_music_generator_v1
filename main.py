import streamlit as st
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"C:\Users\Aaron\anaconda3\data_flair\model_v1.keras")

# Include the note-to-index and index-to-note mappings
index_to_note = {
    0: 'F1', 1: '6.8', 2: '7.0', 3: '0.4.7', 4: '3.8', 5: 'E4', 6: '3.5', 7: 'B-5', 8: 'F6', 9: '11.3.6',
    10: 'G5', 11: '4.7', 12: '5.9.0', 13: '0.2.6', 14: '10.11', 15: '11.2.5.7', 16: '0', 17: 'G3', 18: '8.11',
    19: '10.1', 20: 'B-1', 21: '5.8.11', 22: 'E2', 23: '11.2.4', 24: 'B2', 25: 'F4', 26: '2.6.9', 27: '4.7.10',
    28: 'D6', 29: '6', 30: 'G1', 31: '10.1.5', 32: '5.11', 33: '10.1.4', 34: 'F#3', 35: '2.5', 36: '9.0',
    37: 'C#3', 38: '9.0.2', 39: 'A4', 40: 'C5', 41: '2.5.9', 42: '0.6', 43: '4.7.9', 44: '1.5', 45: '3.6',
    46: 'B-6', 47: 'E5', 48: '11', 49: '5.10', 50: '2.4.8', 51: '3.6.10', 52: '1.4.7', 53: 'C#2', 54: '0.3',
    55: '10.1.3', 56: 'C#5', 57: 'B4', 58: '3.6.9', 59: 'B5', 60: '7.11.2', 61: '2.8', 62: '1.4', 63: 'C#6',
    64: '5', 65: '3.7.10', 66: '11.2', 67: '2.7', 68: 'C2', 69: '4', 70: 'A2', 71: '6.9', 72: 'E6', 73: '8.10',
    74: 'E-6', 75: '9.1.4', 76: '8.11.3', 77: '7', 78: '7.11', 79: '8.0', 80: '6.9.1', 81: '1.3', 82: '0.2',
    83: '6.11', 84: '5.8', 85: 'F#2', 86: '4.10', 87: 'C6', 88: 'G#1', 89: '10.3', 90: '7.9', 91: '1.4.8',
    92: '10.2', 93: '9.2', 94: 'C#4', 95: '9.0.4', 96: '1.7', 97: '8.11.2', 98: '5.7', 99: 'F#6', 100: '0.3.6',
    101: 'F2', 102: 'G#6', 103: '2', 104: '9.1', 105: 'E-5', 106: '3.9', 107: '7.8', 108: '0.3.5', 109: 'F5',
    110: 'E-3', 111: '11.2.6', 112: '5.9', 113: '6.10.1', 114: '4.9', 115: '7.10.0', 116: 'D4', 117: 'D5',
    118: 'G2', 119: '8.0.3', 120: 'G6', 121: '0.1', 122: '10', 123: '3.7', 124: 'A5', 125: '10.2.5', 126: 'A6',
    127: '4.7.11', 128: '7.10.1', 129: 'D3', 130: 'B3', 131: '4.6', 132: '9.10', 133: 'B-4', 134: 'A3', 135: '5.8.0',
    136: '3.4', 137: 'B1', 138: '1.5.8', 139: '4.8.11', 140: 'F3', 141: '4.8', 142: 'D2', 143: '9', 144: 'E3',
    145: '7.9.1', 146: '8.1', 147: '3.6.8', 148: 'F#5', 149: 'C4', 150: 'B-2', 151: '1', 152: '7.10.2', 153: '2.4',
    154: '10.0', 155: 'G#2', 156: 'G#4', 157: 'A1', 158: '0.5', 159: '9.11', 160: '11.4', 161: '11.1', 162: '7.10',
    163: '0.4', 164: 'F#4', 165: '1.4.7.10', 166: '1.6', 167: 'B-3', 168: '11.3', 169: '0.3.7', 170: '2.5.7', 171: '8',
    172: '2.6', 173: 'C7', 174: 'G4', 175: 'G#5', 176: '3', 177: 'E-2', 178: 'C3', 179: '6.10', 180: 'G#3', 181: 'E-4'
}
note_to_index = {v: k for k, v in index_to_note.items()}

# Create a synthetic starting sequence
def create_synthetic_starting_sequence(note_to_index, timesteps=50):
    return np.random.choice(list(note_to_index.values()), timesteps)

# Function to generate music
def generate_music(model, num_notes=200):
    timesteps = 50  # Use the same timestep value as in training
    music_pattern = create_synthetic_starting_sequence(note_to_index, timesteps)
    out_pred = []

    for i in range(num_notes):
        music_pattern = music_pattern.reshape(1, len(music_pattern), 1)
        pred_index = np.argmax(model.predict(music_pattern))
        out_pred.append(index_to_note[pred_index])
        music_pattern = np.append(music_pattern, pred_index)
        music_pattern = music_pattern[1:]

    return out_pred

# Function to save generated notes to a MIDI file
def save_midi(predicted_notes, file_path='output.mid'):
    output_notes = []
    for offset, pattern in enumerate(predicted_notes):
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=file_path)
    return file_path

# Streamlit app
st.set_page_config(page_title="Automatic Music Generator", page_icon=":musical_note:", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🎵 Automatic Music Generator')
st.write('This app generates music using a trained LSTM model. Adjust the settings below and click "Generate Music" to create your own melody!')

# Input: Number of notes to generate
num_notes = st.slider('Number of notes to generate', min_value=50, max_value=500, step=50)

if st.button('Generate Music'):
    st.write('Generating music...')
    predicted_notes = generate_music(model, num_notes)
    midi_file_path = save_midi(predicted_notes)
    st.write('Music generated successfully!')

    # Provide a download link for the MIDI file
    with open(midi_file_path, 'rb') as f:
        st.download_button('Download MIDI file', f, file_name='generated_music.mid')