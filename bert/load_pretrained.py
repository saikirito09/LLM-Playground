import torch
from transformers import BertModel as HuggingFaceBertModel
from model import BertModel, BertConfig

def load_pretrained_weights(custom_model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = custom_model.state_dict()

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict}

    # Handle size mismatch for embeddings if vocab sizes differ
    if custom_dict['embeddings.word_embeddings.weight'].shape != pretrained_dict['embeddings.word_embeddings.weight'].shape:
        custom_dict['embeddings.word_embeddings.weight'][:pretrained_dict['embeddings.word_embeddings.weight'].shape[0]] = pretrained_dict['embeddings.word_embeddings.weight']
        del pretrained_dict['embeddings.word_embeddings.weight']

    # Overwrite entries in the existing state dict
    custom_dict.update(pretrained_dict)

    # Load the new state dict
    custom_model.load_state_dict(custom_dict)

def get_pretrained_bert(model_name='bert-base-uncased'):
    # Load pre-trained model from Hugging Face
    pretrained_model = HuggingFaceBertModel.from_pretrained(model_name)

    # Create a config that matches the pre-trained model
    config = BertConfig()

    # Initialize your custom model
    custom_model = BertModel(config)

    # Load pre-trained weights into your custom model
    load_pretrained_weights(custom_model, pretrained_model)

    return custom_model

if __name__ == "__main__":
    # Example usage
    model = get_pretrained_bert()
    print("Pretrained BERT model loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
