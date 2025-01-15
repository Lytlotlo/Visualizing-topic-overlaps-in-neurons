from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx

# Load a pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input sentence
input_sentence = "Doctors use AI to diagnose diseases."
inputs = tokenizer(input_sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
clean_tokens = [token.replace("Ġ", "") for token in tokens]  # Remove "G" prefix

# Forward pass to get model outputs
outputs = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = outputs.hidden_states  # Tuple of tensors for each layer

# Extract the activation matrix from the last hidden layer
activation_matrix = hidden_states[-2].detach().squeeze(0).numpy()  # Second-to-last layer
sequence_length, hidden_size = activation_matrix.shape

# Simulate weights for final contributions
final_layer_weights = {
    "Health": 0.2,
    "Mixed": 0.3,
    "Technology": 0.5
}

# Neural network visualization setup
layers = [len(clean_tokens), 16, 16, 16, 8, 3, 1]  # [input, hidden1, hidden2, hidden3, hidden4, output]
G = nx.DiGraph()
positions = {}
node_colors = []
edge_colors = []
node_labels = {}

# Adjusted pastel colors
darker_pastel_colors = [
    "#FF9999", "#FFCC99", "#FFFF99", "#99FF99", "#99CCFF", "#CC99FF", "#FF99CC", "#99FFFF"
]
token_colors = darker_pastel_colors[:len(clean_tokens)]

# Define topic colors
topic_colors = {
    "Health": "#FF6666",  # Darker red
    "Mixed": "#FFFF66",   # Darker yellow
    "Technology": "#6699FF",  # Darker blue
}

# Add nodes and edges layer by layer
for layer_idx, num_neurons in enumerate(layers):
    for neuron_idx in range(num_neurons):
        node_id = f"L{layer_idx}_N{neuron_idx}"
        G.add_node(node_id)

        # Position nodes in a grid-like layout
        positions[node_id] = (layer_idx, -neuron_idx)

        # Input layer (tokens)
        if layer_idx == 0:
            node_colors.append(token_colors[neuron_idx])  # Assign unique color to each token
            node_labels[node_id] = clean_tokens[neuron_idx]  # Clean token text
        # Hidden layers
        elif layer_idx < len(layers) - 1:
            # Assign topics and colors to neurons
            contributing_tokens = [clean_tokens[i] for i in range(len(clean_tokens)) if neuron_idx % len(clean_tokens) == i]
            if neuron_idx % 3 == 0:
                topic = "Health"
                node_colors.append(topic_colors[topic])
            elif neuron_idx % 3 == 1:
                topic = "Mixed"
                node_colors.append(topic_colors[topic])
            else:
                topic = "Technology"
                node_colors.append(topic_colors[topic])
            node_labels[node_id] = f"{topic} Neuron {neuron_idx}\nTokens: {', '.join(contributing_tokens)}"
        # Output layer
        else:
            node_colors.append("#FF99CC")  # Light pink for output
            node_labels[node_id] = f"Predicted Topic: Technology"

# Add edges (connections) between layers
for layer_idx in range(len(layers) - 1):
    for src_idx in range(layers[layer_idx]):
        for dst_idx in range(layers[layer_idx + 1]):
            src_node = f"L{layer_idx}_N{src_idx}"
            dst_node = f"L{layer_idx + 1}_N{dst_idx}"

            # Color edges based on token influence (input-to-hidden or hidden-to-hidden)
            if layer_idx == 0:  # Input to first hidden layer
                edge_colors.append(token_colors[src_idx])  # Use token color
            elif layer_idx < len(layers) - 2:  # Hidden-to-hidden layers
                edge_colors.append("#AAAAAA")  # Light gray for neutral edges
            else:  # Last hidden layer to output
                topic = ["Health", "Mixed", "Technology"][src_idx % 3]
                edge_colors.append(topic_colors[topic])  # Match edge color to topic
            G.add_edge(src_node, dst_node)

# Visualize the neural network graph
plt.figure(figsize=(14, 10))
edges = G.edges(data=True)
edge_labels = {
    (src, dst): f"{final_layer_weights.get(topic, 0):.2f}" if 'weight' in data else ""
    for src, dst, data in edges
}
nx.draw(
    G, pos=positions,
    with_labels=True,
    labels=node_labels,
    node_size=600,
    node_color=node_colors,
    edge_color=edge_colors,
    alpha=0.8,
    connectionstyle="arc3,rad=0.2"  # Curved edges for clarity
)

# Annotate the output prediction
plt.text(
    len(layers) - 0.5, -0.5,
    f"Predicted Topic: Technology\nContributions: {final_layer_weights}",
    fontsize=10,
    color="black"
)

# Add legend for token colors and topics
legend_handles = [
    plt.scatter([], [], color=token_colors[i], label=f"Token: {clean_tokens[i]}")
    for i in range(len(clean_tokens))
]
for topic, color in topic_colors.items():
    legend_handles.append(plt.scatter([], [], color=color, label=f"{topic} Contribution"))
legend_handles.append(plt.scatter([], [], color="#FF99CC", label="Output Layer"))
plt.legend(handles=legend_handles, loc="upper right", fontsize=10)

plt.title("Enhanced Token Flow with Darker Pastel Colors and Edge Tracing")
plt.show()


