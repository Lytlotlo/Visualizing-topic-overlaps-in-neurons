from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input sentence
input_sentence = "software will change the way you work"
inputs = tokenizer(input_sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
clean_tokens = [token.replace("Ġ", "") for token in tokens]  # Remove "G" prefix

# Forward pass to get model outputs
outputs = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = outputs.hidden_states  # Tuple of tensors for each layer

# Define topic keywords for semantic similarity
topic_keywords = {
    "Health": "health care, wellness, fitness, medical treatment",
    "Technology": "AI, computers, software, automation, innovation, digital systems",
    "Mixed": "general topics, random discussions, miscellaneous, neutral content",
}

# Get embeddings for the sentence and keywords
sentence_embedding = model.transformer.wte(inputs["input_ids"]).mean(dim=1).detach().numpy()
keyword_embeddings = {
    topic: model.transformer.wte(tokenizer(topic, return_tensors="pt")["input_ids"]).mean(dim=1).detach().numpy()
    for topic in topic_keywords
}

# Compute similarities
similarities = {
    topic: cosine_similarity(sentence_embedding, embedding)[0, 0]
    for topic, embedding in keyword_embeddings.items()
}

# Adjust similarities based on keyword occurrences
sentence_lower = input_sentence.lower()
weights = {
    "Health": 1.0 + 0.5 * any(word in sentence_lower for word in ["health", "care", "medical", "fitness"]),
    "Technology": 1.0 + 0.5 * any(word in sentence_lower for word in ["ai", "software", "automation", "digital"]),
    "Mixed": 1.0,
}
adjusted_similarities = {
    topic: score * weights[topic]
    for topic, score in similarities.items()
}

# Assign topic based on highest similarity
predicted_category = max(adjusted_similarities, key=adjusted_similarities.get)

# Dynamically select 5 hidden layers (instead of 4)
total_hidden_layers = len(hidden_states) - 2  # Exclude input and output layers
selected_layers = np.linspace(0, total_hidden_layers - 1, 5, dtype=int)  # Choose 5 evenly spaced layers
reduced_hidden_states = [hidden_states[i + 1].detach().squeeze(0).numpy() for i in selected_layers]

# Define neurons per layer with at least 3 neurons in the final hidden layer
max_neurons_per_layer = 12
final_hidden_neurons = 3
reduced_activations = [
    layer[:, :max_neurons_per_layer] if i < len(selected_layers) - 1 else layer[:, :final_hidden_neurons]
    for i, layer in enumerate(reduced_hidden_states)
]

# Update the network layers structure to account for the additional hidden layer
layers = [len(clean_tokens)] + [
    min(max_neurons_per_layer, layer.shape[1]) for layer in reduced_activations[:-1]
] + [final_hidden_neurons, 1]

# Neural network visualization setup
G = nx.DiGraph()
positions = {}
node_colors = []
node_labels = []

# Topic colors
topic_colors = {
    "Health": "#FF6666",
    "Technology": "#6699FF",
    "Mixed": "#FFFF66",
}

# Token colors
token_colors = ["#99FF99", "#FFCC99", "#FF9999", "#99CCFF", "#FF99CC"]
while len(token_colors) < len(clean_tokens):
    token_colors.extend(token_colors[:len(clean_tokens) - len(token_colors)])

# Add nodes and define positions based on reduced_activations
for layer_idx, activations in enumerate(reduced_activations):
    num_neurons = activations.shape[1]  # Number of neurons in this layer
    for neuron_idx in range(num_neurons):
        node_id = f"L{layer_idx}_N{neuron_idx}"
        G.add_node(node_id)
         # Add spacing between neurons
        positions[node_id] = (layer_idx, -neuron_idx * 2) 

        # Assign colors and labels
        if layer_idx == 0:  # Input layer
            node_colors.append(token_colors[neuron_idx % len(token_colors)])
            node_labels.append(clean_tokens[neuron_idx % len(clean_tokens)])
        # Hidden layers
        else: 
            topic = list(topic_keywords.keys())[neuron_idx % len(topic_keywords)]  # Assign topics cyclically
            color = topic_colors[topic]
            activation_value = activations[:, neuron_idx].mean()
            node_colors.append(color)
            node_labels.append(f"{topic}\nActivation: {activation_value:.2f}")

# Add output layer node
output_node_id = f"L{len(reduced_activations)}_N0"
G.add_node(output_node_id)
positions[output_node_id] = (len(reduced_activations), 0)  # Single output node
node_colors.append("#FF99CC")  # Output layer color
node_labels.append(f"Prediction:\n{predicted_category}")

# Add cooperation edges
cooperation_threshold = 0.2
for layer_idx in range(len(reduced_activations) - 1):
    for src_idx in range(reduced_activations[layer_idx].shape[1]):
        for dst_idx in range(reduced_activations[layer_idx + 1].shape[1]):
            cooperation_strength = sum(
                reduced_activations[layer_idx][:, src_idx] *
                reduced_activations[layer_idx + 1][:, dst_idx]
            )
            if cooperation_strength > cooperation_threshold:
                src_node = f"L{layer_idx}_N{src_idx}"
                dst_node = f"L{layer_idx + 1}_N{dst_idx}"
                G.add_edge(src_node, dst_node, weight=cooperation_strength)

# Adjust edge colors and thickness
edges = G.edges(data=True)
edge_weights = [data["weight"] if "weight" in data else 0 for _, _, data in edges]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [2 + (weight / max_weight) * 8 for weight in edge_weights]

# Plot graph with Axes
fig, ax = plt.subplots(figsize=(14, 10))
nx.draw(
    G, pos=positions,
    with_labels=True,
    labels=dict(zip(G.nodes(), node_labels)),
    node_color=node_colors,
    node_size=600,
    edge_color=edge_weights,
    edge_cmap=plt.cm.Blues,
    width=edge_widths,
    alpha=0.8,
    connectionstyle="arc3,rad=0.2",
    ax=ax
)

# Add colorbar for cooperation strength
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_weight))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Cooperation Strength")

# Add legend
legend_handles = [plt.scatter([], [], color=color, label=f"{topic}") for topic, color in topic_colors.items()]
legend_handles.append(plt.scatter([], [], color="#FF99CC", label="Output Layer"))
ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

ax.set_title("Neural Network Visualization with Cooperation Highlighting (5 Hidden Layers)")
plt.show()

