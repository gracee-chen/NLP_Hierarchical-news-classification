import streamlit as st
import torch
import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data import load_level1_df, load_level2_df, make_label_maps

# Page setup
st.set_page_config(page_title="NLP Text Classification Demo", layout="wide")
st.title("NLP Text Classification Demo")

# Load models and tokenizers
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load level 1 classification model
    l1_model_path = os.path.join(current_dir, "outputs/l1/best")
    
    # Load label mappings
    l1_df = load_level1_df()
    l1_label2id, l1_id2label = make_label_maps(l1_df["level1"])
    
    # Load level 2 classification models (optional)
    l2_models = {}
    l2_id2labels = {}
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        l1_model = AutoModelForSequenceClassification.from_pretrained(
            l1_model_path,
            local_files_only=True
        )
        
        # Try loading level 2 classification models
        l2_df = load_level2_df()
        
        # Check what directories actually exist in outputs/l2
        l2_base_path = os.path.join(current_dir, "outputs/l2")
        if os.path.exists(l2_base_path):
            existing_dirs = os.listdir(l2_base_path)
            # st.write("Found level 2 directories:", existing_dirs)
            
            for parent in l1_id2label.values():
                # Check if exact directory exists
                if parent in existing_dirs:
                    dir_name = parent
                else:
                    # Try to find a matching directory
                    matching_dirs = [d for d in existing_dirs if d.replace("_", " ").lower() == parent.lower()]
                    if matching_dirs:
                        dir_name = matching_dirs[0]
                    else:
                        # Skip this category if no matching directory found
                        # st.info(f"No directory found for category: {parent}")
                        continue
                
                l2_path = os.path.join(l2_base_path, dir_name)
                
                # Check if we need to look in a "best" subdirectory
                best_path = os.path.join(l2_path, "best")
                if os.path.exists(best_path):
                    l2_path = best_path
                
                try:
                    # Check if config.json exists and has model_type
                    config_path = os.path.join(l2_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                            if 'model_type' not in config_data:
                                # Add model_type to config
                                config_data['model_type'] = 'bert'
                                with open(config_path, 'w') as f:
                                    json.dump(config_data, f)
                    
                    # Now try to load the model
                    l2_model = AutoModelForSequenceClassification.from_pretrained(
                        l2_path,
                        local_files_only=True
                    )
                    
                    # Get level 2 labels for this parent category
                    parent_df = l2_df[l2_df["level1"] == parent]
                    l2_label2id, l2_id2label = make_label_maps(parent_df["level2"])
                    
                    l2_models[parent] = l2_model
                    l2_id2labels[parent] = l2_id2label
                    # st.success(f"Successfully loaded level 2 model for {parent}")
                except Exception as e:
                    # st.warning(f"Unable to load level 2 classification model for {parent}: {str(e)}")
                    pass
        
        return tokenizer, l1_model, l1_id2label, l2_models, l2_id2labels
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Load models
tokenizer, l1_model, l1_id2label, l2_models, l2_id2labels = load_models()

# st.write(f"Loaded {len(l2_models)} level 2 models for categories: {list(l2_models.keys())}")

# Continue only if models loaded successfully
if tokenizer is not None and l1_model is not None:
    # Create input area
    user_input = st.text_area("Enter text for classification:", height=150)

    if st.button("Classify"):
        if user_input:
            # Text preprocessing and classification
            with st.spinner("Processing..."):
                inputs = tokenizer(
                    user_input, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=512
                )
                
                # Level 1 classification
                with torch.no_grad():
                    l1_outputs = l1_model(**inputs)
                    l1_predictions = torch.nn.functional.softmax(l1_outputs.logits, dim=-1)
                    l1_pred_class = torch.argmax(l1_predictions, dim=-1).item()
                    l1_label = l1_id2label[l1_pred_class]
                    l1_confidence = l1_predictions[0][l1_pred_class].item()
                
                # Try level 2 classification
                l2_label = None
                l2_confidence = None
                l2_predictions = None
                
                if l1_label in l2_models:
                    l2_model = l2_models[l1_label]
                    l2_id2label = l2_id2labels[l1_label]
                    
                    with torch.no_grad():
                        l2_outputs = l2_model(**inputs)
                        l2_predictions = torch.nn.functional.softmax(l2_outputs.logits, dim=-1)
                        l2_pred_class = torch.argmax(l2_predictions, dim=-1).item()
                        l2_label = l2_id2label[l2_pred_class]
                        l2_confidence = l2_predictions[0][l2_pred_class].item()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Level 1 Classification Results")
                st.success(f"Predicted Category: {l1_label}")
                st.info(f"Confidence: {l1_confidence:.2%}")
                
                # Display confidence for all level 1 categories
                st.markdown("Confidence for all Level 1 categories:")
                l1_probs_dict = {l1_id2label[i]: l1_predictions[0][i].item() for i in range(len(l1_id2label))}
                st.bar_chart(l1_probs_dict)
            
            with col2:
                if l2_label:
                    st.subheader("Level 2 Classification Results")
                    st.success(f"Predicted Category: {l2_label}")
                    st.info(f"Confidence: {l2_confidence:.2%}")
                    
                    # Display confidence for all level 2 categories
                    st.markdown("Confidence for all Level 2 categories:")
                    l2_probs_dict = {l2_id2labels[l1_label][i]: l2_predictions[0][i].item() 
                                    for i in range(len(l2_id2labels[l1_label]))}
                    st.bar_chart(l2_probs_dict)
                else:
                    st.warning(f"No Level 2 classification model found for '{l1_label}'")
        else:
            st.warning("Please enter text for classification")

    # Add instructions
    st.markdown("""
    ### Instructions
    1. Enter the text you want to classify in the text box
    2. Click the "Classify" button
    3. View the prediction results and confidence levels
    """)
else:
    st.error("Unable to load models. Please check model paths and configuration.")