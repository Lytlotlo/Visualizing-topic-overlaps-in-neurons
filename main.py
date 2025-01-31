from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ensure the 'static' folder exists
os.makedirs("static", exist_ok=True)

# Load a pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input sentence
input_sentence = "software will change the way you work"
inputs = tokenizer(input_sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
clean_tokens = [token.replace("Ġ", "") for token in tokens]  # Remove "Ġ" prefix

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
total_hidden_layers = len(hidden_states) - 2  # Exclude input & final output layers from GPT-2
selected_layers = np.linspace(0, total_hidden_layers - 1, 5, dtype=int)  # Choose 5 evenly spaced layers

# Collect embeddings for those 5 layers
reduced_hidden_states = [hidden_states[i + 1].detach().squeeze(0).numpy() for i in selected_layers]

# Define neurons per layer with at least 3 neurons in the final hidden layer
max_neurons_per_layer = 12
final_hidden_neurons = 3

reduced_activations = []
for i, layer in enumerate(reduced_hidden_states):
    if i < len(reduced_hidden_states) - 1:
        # Limit hidden layers to max_neurons_per_layer
        reduced_activations.append(layer[:, :max_neurons_per_layer])
    else:
        # Final hidden layer to final_hidden_neurons
        reduced_activations.append(layer[:, :final_hidden_neurons])

# Create a separate input layer for token embeddings
# Shape: [sequence_length, hidden_dim]. We slice to match the actual token count.
input_layer_activations = model.transformer.wte(inputs["input_ids"]).detach().squeeze(0).numpy()
num_actual_tokens = len(clean_tokens)
input_layer_activations = input_layer_activations[:num_actual_tokens, :]

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
    token_colors.extend(token_colors[: len(clean_tokens) - len(token_colors)])

# 1) Add INPUT layer nodes (tokens) -- now the length matches clean_tokens
for neuron_idx in range(num_actual_tokens):
    node_id = f"L0_N{neuron_idx}"
    G.add_node(node_id)
    positions[node_id] = (0, -neuron_idx * 2)
    node_colors.append(token_colors[neuron_idx % len(token_colors)])
    node_labels.append(clean_tokens[neuron_idx])

# 2) Add the 5 hidden layers
for layer_idx, activations in enumerate(reduced_activations, start=1):
    num_neurons = activations.shape[1]
    for neuron_idx in range(num_neurons):
        node_id = f"L{layer_idx}_N{neuron_idx}"
        G.add_node(node_id)
        positions[node_id] = (layer_idx, -neuron_idx * 2)

        # Assign color/label
        topic = list(topic_colors.keys())[neuron_idx % len(topic_colors)]
        color = topic_colors[topic]
        activation_value = activations[:, neuron_idx].mean()
        node_colors.append(color)
        node_labels.append(f"{topic}\nActivation: {activation_value:.2f}")

# 3) Add OUTPUT layer node (L6_N0)
output_node_id = f"L{len(reduced_activations)+1}_N0"
G.add_node(output_node_id)
positions[output_node_id] = (len(reduced_activations) + 1, 0)
node_colors.append("#FF99CC")
node_labels.append(f"Prediction:\n{predicted_category}")

# 4) Create edges
def compute_cooperation_strength(layerA, layerB, src_idx, dst_idx):
    return float(np.sum(layerA[:, src_idx] * layerB[:, dst_idx]))

cooperation_threshold = 0.2
input_cooperation_threshold = 0.0

# 4a) INPUT -> first hidden layer
first_hidden = reduced_activations[0]
for src_idx in range(num_actual_tokens):
    for dst_idx in range(first_hidden.shape[1]):
        strength = compute_cooperation_strength(input_layer_activations, first_hidden, src_idx, dst_idx)
        if strength > input_cooperation_threshold:
            G.add_edge(f"L0_N{src_idx}", f"L1_N{dst_idx}", weight=strength)

# 4b) Hidden layer -> hidden layer
for layer_idx in range(len(reduced_activations) - 1):
    layerA = reduced_activations[layer_idx]
    layerB = reduced_activations[layer_idx + 1]
    for src_idx in range(layerA.shape[1]):
        for dst_idx in range(layerB.shape[1]):
            strength = compute_cooperation_strength(layerA, layerB, src_idx, dst_idx)
            if strength > cooperation_threshold:
                G.add_edge(f"L{layer_idx+1}_N{src_idx}", f"L{layer_idx+2}_N{dst_idx}", weight=strength)

# 4c) Last hidden -> OUTPUT
last_hidden = reduced_activations[-1]
for src_idx in range(last_hidden.shape[1]):
    strength = float(np.sum(last_hidden[:, src_idx]))
    if abs(strength) > 0.0:
        G.add_edge(f"L{len(reduced_activations)}_N{src_idx}", output_node_id, weight=strength)

# 5) Plot (increased figure size)
fig, ax = plt.subplots(figsize=(18, 12))

edges = G.edges(data=True)
edge_weights = [data["weight"] if "weight" in data else 0 for _, _, data in edges]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [2 + (w / max_weight) * 8 for w in edge_weights]

nx.draw(
    G,
    pos=positions,
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

# Colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_weight))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Cooperation Strength")

# Legend
legend_handles = [
    plt.scatter([], [], color=color, label=topic)
    for topic, color in topic_colors.items()
]
legend_handles.append(plt.scatter([], [], color="#FF99CC", label="Output Layer"))
ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

ax.set_title("Neural Network Visualization with Cooperation Highlighting (5 Hidden Layers)")

# Save the plot
output_path = "static/graph.png"
fig.savefig(output_path, format="png", bbox_inches="tight")
print(f"Graph saved to {output_path}")
plt.close(fig)

