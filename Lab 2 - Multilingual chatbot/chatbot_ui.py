"""
Riva RAG Chatbot UI Module

This module provides the interactive UI for the Riva RAG Chatbot Lab.
It handles audio recording via browser microphone, file uploads, and manages the chat interaction flow.
"""

import base64
import os
import ipywidgets as widgets
from IPython.display import display, Javascript


# Input directory for audio files
INPUT_DIR = "./input"
os.makedirs(INPUT_DIR, exist_ok=True)

# Language code to name mapping (only supported languages)
LANGUAGE_NAMES = {
    'en-US': 'English',
    'es-US': 'Spanish',
    'fr-FR': 'French'
}


def create_chatbot_ui(transcribe_fn, query_rag_fn, speak_fn):
    """
    Create an interactive chatbot UI with audio recording and file upload capabilities.
    
    Args:
        transcribe_fn: Function to transcribe audio (audio_file) -> str
        query_rag_fn: Function to query RAG endpoint (text, language) -> str
        speak_fn: Function to synthesize speech (text, language_code) -> None
    
    Returns:
        None (displays the UI directly)
    """
    # Header
    header = widgets.HTML("<h2>Riva RAG Chatbot</h2>")
    
    # Language Selection (only supported languages)
    lang_dropdown = widgets.Dropdown(
        options=[
            ('English (US)', 'en-US'),
            ('Spanish (US)', 'es-US'),
            ('French', 'fr-FR')
        ],
        value='en-US',
        description='Output Language:',
        style={'description_width': 'initial'}
    )
    
    # Recording state
    is_recording = {'value': False}
    
    # Start/Stop Recording Buttons
    start_btn = widgets.Button(
        description='Start Recording',
        button_style='success',
        layout=widgets.Layout(width='150px'),
        icon='microphone'
    )
    
    stop_btn = widgets.Button(
        description='Stop Recording',
        button_style='danger',
        layout=widgets.Layout(width='150px'),
        icon='stop',
        disabled=True
    )
    
    # File Upload
    upload_btn = widgets.FileUpload(
        accept='.wav,.mp3',
        multiple=False,
        description='Upload Audio',
        button_style='info',
        layout=widgets.Layout(width='150px')
    )
    
    # File Selector for existing files in input directory
    def get_audio_files():
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.wav', '.mp3'))]
        return [('-- Select a file --', '')] + [(f, f) for f in sorted(files)]
    
    file_selector = widgets.Dropdown(
        options=get_audio_files(),
        description='Saved Files:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    refresh_btn = widgets.Button(
        description='Refresh',
        button_style='',
        layout=widgets.Layout(width='100px'),
        icon='refresh'
    )
    
    process_file_btn = widgets.Button(
        description='Process Selected',
        button_style='primary',
        layout=widgets.Layout(width='150px'),
        disabled=True
    )
    
    # Hidden widget to transfer audio data from JS to Python
    audio_data_widget = widgets.Textarea(value='', layout=widgets.Layout(display='none'))
    audio_data_widget.add_class('audio-data-transfer')
    
    # Output Area
    out = widgets.Output(layout={'border': '1px solid #ccc', 'padding': '10px', 'min_height': '200px'})
    
    # Javascript for Recording (WAV Encoder 16kHz Mono)
    start_recording_js = """
    window.audioRecorder = {
        stream: null,
        context: null,
        processor: null,
        source: null,
        samples: []
    };
    
    async function startRecording() {
        const btn = document.querySelector('.jupyter-button.mod-success');
        if(btn) btn.style.backgroundColor = '#ff4444';
        
        try {
            window.audioRecorder.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            window.audioRecorder.context = new AudioContext({ sampleRate: 16000 });
            window.audioRecorder.source = window.audioRecorder.context.createMediaStreamSource(window.audioRecorder.stream);
            window.audioRecorder.processor = window.audioRecorder.context.createScriptProcessor(4096, 1, 1);
            
            window.audioRecorder.samples = [];
            window.audioRecorder.processor.onaudioprocess = (e) => {
                const input = e.inputBuffer.getChannelData(0);
                for (let i = 0; i < input.length; i++) {
                    window.audioRecorder.samples.push(input[i]);
                }
            };
            
            window.audioRecorder.source.connect(window.audioRecorder.processor);
            window.audioRecorder.processor.connect(window.audioRecorder.context.destination);
            
        } catch(err) {
            console.error(err);
            alert("Error starting recording: " + err);
            if(btn) btn.style.backgroundColor = '';
        }
    }
    startRecording();
    """
    
    stop_recording_js = """
    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);
        
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            s = s < 0 ? s * 0x8000 : s * 0x7FFF;
            view.setInt16(offset, s, true);
            offset += 2;
        }
        return new Blob([view], { type: 'audio/wav' });
    }
    
    async function stopRecording() {
        const btn = document.querySelector('.jupyter-button.mod-success');
        
        if (window.audioRecorder.source) {
            window.audioRecorder.source.disconnect();
            window.audioRecorder.processor.disconnect();
            window.audioRecorder.context.close();
            
            const wavBlob = encodeWAV(window.audioRecorder.samples, 16000);
            
            const reader = new FileReader();
            reader.readAsDataURL(wavBlob);
            reader.onloadend = () => {
                const base64data = reader.result.split(',')[1];
                const textareas = document.getElementsByClassName('audio-data-transfer');
                if (textareas.length > 0) {
                    const textarea = textareas[0].querySelector('textarea');
                    if (textarea) {
                        textarea.value = base64data;
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }
                if(btn) btn.style.backgroundColor = '';
            };
        }
    }
    stopRecording();
    """
    
    def on_start_click(b):
        is_recording['value'] = True
        start_btn.disabled = True
        stop_btn.disabled = False
        with out:
            print("Recording started... Click 'Stop Recording' when done.")
            display(Javascript(start_recording_js))
    
    def on_stop_click(b):
        is_recording['value'] = False
        start_btn.disabled = False
        stop_btn.disabled = True
        audio_data_widget.value = ""
        with out:
            print("Stopping recording...")
            display(Javascript(stop_recording_js))

    def on_audio_received(change):
        if not change['new']:
            return
            
        b64_data = change['new']
        timestamp = int(os.times().elapsed * 1000)
        filename = os.path.join(INPUT_DIR, f"recorded_{timestamp}.wav")
        
        with out:
            print("Audio captured. Saving...")
            try:
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                
                print(f"Saved to: {filename}")
                process_interaction(filename, lang_dropdown.value)
                
                # Refresh file list
                file_selector.options = get_audio_files()
                
            except Exception as e:
                print(f"Error saving/processing audio: {e}")
    
    def on_upload_change(change):
        if not upload_btn.value:
            return

        # Handle different ipywidgets versions
        # ipywidgets 7: value is a dict {filename: {metadata: {...}, content: ...}}
        # ipywidgets 8: value is a tuple ({name: ..., content: ...}, ...)
        
        try:
            if isinstance(upload_btn.value, tuple):
                # ipywidgets 8+
                uploaded_file = upload_btn.value[0]
                filename = uploaded_file['name']
                content = uploaded_file['content']
                # content might be memoryview
                if hasattr(content, 'tobytes'):
                    content = content.tobytes()
            else:
                # ipywidgets 7
                uploaded_file = list(upload_btn.value.values())[0]
                filename = uploaded_file['metadata']['name']
                content = uploaded_file['content']

            filepath = os.path.join(INPUT_DIR, filename)
            
            with out:
                out.clear_output()
                print(f"Uploading file: {filename}")
                try:
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    
                    print(f"Saved to: {filepath}")
                    process_interaction(filepath, lang_dropdown.value)
                    
                    # Refresh file list
                    file_selector.options = get_audio_files()
                    
                except Exception as e:
                    print(f"Error processing uploaded file: {e}")
        except Exception as e:
            with out:
                print(f"Error handling upload: {e}")
    
    def on_refresh_click(b):
        file_selector.options = get_audio_files()
        with out:
            out.clear_output()
            print("File list refreshed!")
    
    def on_file_select_change(change):
        process_file_btn.disabled = (change['new'] == '')
    
    def on_process_file_click(b):
        if file_selector.value:
            filename = os.path.join(INPUT_DIR, file_selector.value)
            with out:
                out.clear_output()
                print(f"Processing file: {file_selector.value}")
                process_interaction(filename, lang_dropdown.value)

    def process_interaction(audio_file, lang_code):
        # Get language name from code
        language_name = LANGUAGE_NAMES.get(lang_code, 'English')
        
        # 1. Transcribe (auto-detect language)
        print("\nTranscribing...")
        text = transcribe_fn(audio_file)
        print(f"You asked: {text}")
        
        if text:
            # 2. RAG Query (with language instruction)
            print(f"Querying Knowledge Base (requesting {language_name} response)...")
            answer = query_rag_fn(text, language_name)
            print(f"Bot Answer: {answer}")
            
            # 3. TTS (use selected language)
            print(f"Speaking answer in {language_name}...")
            speak_fn(answer, lang_code)
        else:
            print("Could not understand the audio. Please try again.")

    # Wire up events
    start_btn.on_click(on_start_click)
    stop_btn.on_click(on_stop_click)
    audio_data_widget.observe(on_audio_received, names='value')
    upload_btn.observe(on_upload_change, names='value')
    refresh_btn.on_click(on_refresh_click)
    file_selector.observe(on_file_select_change, names='value')
    process_file_btn.on_click(on_process_file_click)
    
    # Layout
    recording_box = widgets.HBox([start_btn, stop_btn])
    file_box = widgets.HBox([file_selector, refresh_btn, process_file_btn])
    
    # Display UI
    display(header)
    display(widgets.HTML("<p><strong>Note:</strong> TTS supports English, Spanish, and French only.</p>"))
    display(lang_dropdown)
    display(widgets.HTML("<h3>Record Audio</h3>"))
    display(recording_box)
    display(widgets.HTML("<h3>Upload Audio File</h3><p><strong>Note:</strong> only WAV Format is supported .</p>"))
    display(upload_btn)
    display(widgets.HTML("<h3>Or Select Saved File</h3>"))
    display(file_box)
    display(audio_data_widget)
    display(out)
