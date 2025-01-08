from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import matplotlib.pyplot as plt


# Load a pre-trained model and tokenizer
model_name = "gpt2"
# Loads the GPT-2 model from Hugging Face's pre-trained models.
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
# Loads the tokenizer for GPT-2. The tokenizer will: 
# Break input texts into smaller pieces and convert those 
    # into numerical IDs.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input data
input_text = "AI is transforming healthcare."
#Tokenizes the input into a format the model can understand.
inputs = tokenizer(input_text, return_tensors="pt")


# Forward pass to get model outputs
# Passes the tokenized input through the model to get its outputs
outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# Extract hidden states
hidden_states = outputs.hidden_states  # A tuple of tensors for each layer

# Example: Print the activations of the last layer
last_layer_activations = hidden_states[-1]  # Last layer

# This tells us the shape of the activations.
print(last_layer_activations.shape)  # Shape: [batch_size, sequence_length, hidden_size]


# ANALYZE NEURON ACTIVATIONS FOR A SPECIFIC TOKEN
# Choose a token to analyze (e.g., the first token, "AI")
token_index = 0  # Index of the token in the sequence
activations_for_token = last_layer_activations[0, token_index, :]  # Shape: [768]

# Find the most strongly activated neurons
strongest_neurons = torch.topk(activations_for_token, k=10)  # Top 10 strongest activations
print("Top 10 neuron activations:", strongest_neurons.values)
print("Neuron indices:", strongest_neurons.indices)


# MAP ACTIVATIONS TO TOPICS
# Input sentences for analysis
sentences = ["AI is transforming healthcare.",  # Mixed
             "Hospitals are improving healthcare.",  # Health
             "AI algorithms are advancing quickly."]  # Technology

# Analyze activations for each sentence
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_layer_activations = outputs.hidden_states[-1]  # Last layer

    # Average activations across all tokens in the sentence
    avg_activations = last_layer_activations.mean(dim=1)  # Shape: [batch_size, hidden_size]
    print(f"Average activations for '{sentence}':")
    print(avg_activations)


# VISUALIZE NEURON ACTIVATIONS
# Plot the activations for a specific token
plt.bar(range(10), strongest_neurons.values.detach().numpy())
plt.xlabel("Neuron Index (Top 10)")
plt.ylabel("Activation Value")
plt.title(f"Top 10 Neuron Activations for Token at Index {token_index}")
plt.show()