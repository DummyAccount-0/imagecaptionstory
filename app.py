import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

# Streamlit configuration
st.set_page_config(page_title="Image Caption Generator", layout="wide")

# Load Pre-trained Models
@st.cache_resource
def load_models():
    # Image captioning model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()

    # Story generation model (GPT-2)
    story_model = GPT2LMHeadModel.from_pretrained("gpt2")
    story_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    story_model.eval()

    return processor, model, story_model, story_tokenizer

with st.spinner('Loading models, please wait...'):
    processor, model, story_model, story_tokenizer = load_models()

# Preprocess Image
def preprocess_image(image):
    return image.convert("RGB")

# Generate Caption using BLIP
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Generate Story
def generate_story(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.95):
    inputs = story_tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = story_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=temperature, top_k=top_k, top_p=top_p)
    story = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# Streamlit UI
st.title("🖼️ Image Caption Generator")
st.markdown("""
Choose whether to upload an image or capture one using your camera.
Our AI will generate a caption for your image and a story based on the caption.
""")

with st.sidebar:
    st.header("Choose Image Source")
    option = st.radio("Choose Image Source", ("Upload an Image", "Capture Image from Camera"))

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'caption' not in st.session_state:
        st.session_state.caption = None
    if 'story' not in st.session_state:
        st.session_state.story = None

    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    elif option == "Capture Image from Camera":
        uploaded_file = st.camera_input("Capture image from camera")

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image" if option == "Upload an Image" else "Captured Image", width=400)

with st.container():
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button('Generate Caption'):
            if uploaded_file:
                with st.spinner('Generating caption...'):
                    initial_caption = generate_caption(image)
                    st.session_state.caption = initial_caption
                    st.success("Caption Generated 🎉")
            else:
                st.warning("Please upload or capture an image first.")

    with col2:
        if st.button('Generate Story'):
            if uploaded_file and st.session_state.caption:
                with st.spinner('Generating story...'):
                    story_prompt = f"Once upon a time, {st.session_state.caption}."
                    st.session_state.story = generate_story(story_prompt)
                    st.success("Story Generated 🎉")
            else:
                st.warning("Please generate a caption first.")

    st.markdown("---")
    if st.session_state.caption:
        with st.container():
            st.subheader("Generated Caption")
            st.markdown(f"**Caption:** {st.session_state.caption}")

    if st.session_state.story:
        with st.container():
            st.subheader("Generated Story")
            st.markdown(f"**Story:** {st.session_state.story}")

    if uploaded_file is None:
        st.warning("No image uploaded or captured. Please upload or capture an image to continue.")

st.markdown("""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #000;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
    ">
    Made with ❤ by Rajat Prakash Dhal 
    </div>
""", unsafe_allow_html=True)
