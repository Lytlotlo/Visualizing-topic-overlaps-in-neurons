from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Load a pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define topic-specific sentences
sentences = [
    "Doctors use AI to diagnose diseases.",  # Health
    "AI is revolutionizing the medical field.",  # Mixed
    "AI algorithms are advancing rapidly."  # Technology
]

# Collect activations
all_activations = []
tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]  # Tokenized words

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_layer_activations = outputs.hidden_states[-1]  # Last hidden layer

    # Average activations across all tokens
    avg_activations = last_layer_activations.mean(dim=1)
    all_activations.append(avg_activations.detach().numpy().squeeze())

# Convert activations into a matrix
activation_matrix = np.stack(all_activations)
print("Activation Matrix Shape:", activation_matrix.shape)  # [num_sentences, hidden_size]

# Calculate topic contributions for each neuron
topic_contributions = {}

for neuron_id in range(activation_matrix.shape[1]):  # Iterate over neurons
    neuron_activations = activation_matrix[:, neuron_id]  # Activations for this neuron
    topic_contributions[neuron_id] = neuron_activations / neuron_activations.sum()  # Normalize

# Normalize contributions to ensure they fall within the 0-1 range
def get_color(contributions):
    r = max(0, min(1, contributions[0]))  # Health (Red)
    g = max(0, min(1, contributions[1]))  # Mixed (Green)
    b = max(0, min(1, contributions[2]))  # Technology (Blue)
    return (r, g, b)  # Ensure all values are within [0, 1]

# Neural network visualization setup
layers = [len(sentences), 16, 12, 8, 3]  # Example: [input, hidden1, hidden2, output]

# Generate a neural network graph
G = nx.DiGraph()  # Directed graph
positions = {}  # To store node positions for visualization
node_colors = []  # To store colors for each node
node_labels = {}  # Labels for nodes

# Add nodes for each layer
for layer_idx, num_neurons in enumerate(layers):
    for neuron_idx in range(num_neurons):
        node_id = f"L{layer_idx}_N{neuron_idx}"
        G.add_node(node_id)

        # Position nodes in a grid-like layout
        positions[node_id] = (layer_idx, -neuron_idx)

        # Assign colors based on topic contributions for hidden/output layers
        if layer_idx == 0:  # Input layer
            node_colors.append("green")  # Input layer (green for sentences)
            node_labels[node_id] = tokenized_sentences[neuron_idx][0]  # Display the first word for each sentence
        elif layer_idx == len(layers) - 1:  # Output layer
            node_colors.append("red")  # Output layer (red)
            node_labels[node_id] = "Output"  # Label output layer nodes
        else:
            # Hidden layers: Use topic contributions or random colors
            if neuron_idx < len(topic_contributions):  # Ensure neurons exist in data
                node_colors.append(get_color(topic_contributions.get(neuron_idx, [0.5, 0.5, 0.5])))
                topic_label = ["Health", "Mixed", "Technology"][
                    np.argmax(topic_contributions.get(neuron_idx, [0, 0, 0]))
                ]
                node_labels[node_id] = topic_label  # Label hidden neurons by dominant topic
            else:
                node_colors.append((0.7, 0.7, 0.7))  # Default gray for unused neurons
                node_labels[node_id] = "Unknown"

# Add edges (connections) between layers
for layer_idx in range(len(layers) - 1):
    for src_idx in range(layers[layer_idx]):
        for dst_idx in range(layers[layer_idx + 1]):
            src_node = f"L{layer_idx}_N{src_idx}"
            dst_node = f"L{layer_idx + 1}_N{dst_idx}"
            G.add_edge(src_node, dst_node)

# Visualize the neural network graph
plt.figure(figsize=(14, 10))
nx.draw(
    G, pos=positions,
    with_labels=True,  # Display node labels
    labels=node_labels,  # Add custom labels
    node_size=500,      # Adjust node size
    node_color=node_colors,  # Use assigned colors
    edge_color="gray",   # Connection color
    alpha=0.8            # Transparency for connections
)

# Add legend for the colors
plt.scatter([], [], color="green", label="Input Layer (Sentences)")
plt.scatter([], [], color="red", label="Output Layer (Predictions)")
plt.scatter([], [], color="blue", label="Technology (Neuron Contribution)")
plt.scatter([], [], color="orange", label="Mixed Topics (Neuron Contribution)")
plt.scatter([], [], color="purple", label="Health (Neuron Contribution)")
plt.legend(loc="upper right", fontsize=10)

plt.title("Detailed Neural Network Diagram with Word and Topic Associations")
plt.show()


