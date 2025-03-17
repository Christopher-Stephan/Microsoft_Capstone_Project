import os
import warnings

# Workaround for TensorFlow / protobuf issue (if needed)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Suppress the specific warning about unused weights
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at")

from transformers import DistilBertTokenizer, DistilBertModel
import torch

def main():
    print("Loading DistilBERT tokenizer and model...")
    # Load the tokenizer and model from the Hugging Face hub
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    # (Optional) Set the model to evaluation mode if you're only doing inference:
    model.eval()
    
    # Example text
    text = "This is an example sentence for DistilBERT."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print out the shape of the last hidden state
    print("Output shape:", outputs.last_hidden_state.shape)
    
    # (Optional) Save the model checkpoint to a pickle file for later use in your Streamlit app.
    # For example, you might save just the state_dict.
    save_path = os.path.join(os.path.dirname(__file__), "nrms_bert_state_dict.pkl")
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to: {save_path}")

if __name__ == "__main__":
    main()
