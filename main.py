import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure 'static' exists for saving the final plot
os.makedirs("static", exist_ok=True)

# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input sentence
input_text = "software will change the way you work"
input_ids = tokenizer(input_text, return_tensors="pt")
token_list = tokenizer.convert_ids_to_tokens(input_ids["input_ids"].squeeze())

# Remove the special GPT-2 prefix from tokens
processed_tokens = [tok.replace("Ġ", "") for tok in token_list]

# Forward pass to extract hidden states
model_outputs = model(**input_ids, return_dict=True, output_hidden_states=True)
states = model_outputs.hidden_states  # tuple of hidden state tensors

# Define semantic categories for topic detection
semantic_categories = {
    "Health": "health care, wellness, fitness, medical treatment",
    "Technology": "AI, computers, software, automation, innovation, digital systems",
    "Mixed": "general topics, random discussions, miscellaneous, neutral content",
}

# Compute an embedding for the input text
text_embedding = model.transformer.wte(input_ids["input_ids"]).mean(dim=1).detach().numpy()

# Compute average embeddings for each category
cat_embeddings = {}
for label, desc in semantic_categories.items():
    cat_input = tokenizer(desc, return_tensors="pt")["input_ids"]
    cat_embed = model.transformer.wte(cat_input).mean(dim=1).detach().numpy()
    cat_embeddings[label] = cat_embed

# Measure similarity between text and each category
raw_similarities = {}
for label, embed in cat_embeddings.items():
    raw_similarities[label] = cosine_similarity(text_embedding, embed)[0, 0]

# If certain keywords appear, add weighting to that category’s similarity
input_lower = input_text.lower()
boosts = {
    "Health": 1.0 + 0.5 * any(k in input_lower for k in ["health", "care", "medical", "fitness"]),
    "Technology": 1.0 + 0.5 * any(k in input_lower for k in ["ai", "software", "automation", "digital"]),
    "Mixed": 1.0
}
weighted_similarities = {k: raw_similarities[k] * boosts[k] for k in raw_similarities}

# Choose the category with the highest similarity
topic_assigned = max(weighted_similarities, key=weighted_similarities.get)

# Decide which hidden layers to visualize (picking 5 out of the total)
num_layers = len(states) - 2  # skip input & final output layers
layer_indices = np.linspace(0, num_layers - 1, 5, dtype=int)

# Extract the chosen layers (remove batch dimension)
chosen_layers = [states[idx + 1].detach().squeeze(0).numpy() for idx in layer_indices]

# Limit neurons displayed
max_neurons = 12
final_neurons = 3
activation_slices = []
for i, layer_array in enumerate(chosen_layers):
    if i < len(chosen_layers) - 1:
        activation_slices.append(layer_array[:, :max_neurons])
    else:
        activation_slices.append(layer_array[:, :final_neurons])

# Also keep track of token embeddings for the input layer
input_embeds = model.transformer.wte(input_ids["input_ids"]).detach().squeeze(0).numpy()
num_tokens = len(processed_tokens)
input_embeds = input_embeds[:num_tokens, :]

# Build the network graph
G = nx.DiGraph()
positions = {}
node_labels = {}
node_colors = {}
node_activ_values = {}

# Assign colors for categories
category_colors = {
    "Health": "#FF6666",
    "Technology": "#6699FF",
    "Mixed": "#FFFF66",
}

# Colors for input tokens
token_palette = ["#99FF99", "#FFCC99", "#FF9999", "#99CCFF", "#FF99CC"]
while len(token_palette) < len(processed_tokens):
    token_palette.extend(token_palette[: len(processed_tokens) - len(token_palette)])

# 1) Input layer
for i in range(num_tokens):
    node_id = f"Input_{i}"
    G.add_node(node_id)
    positions[node_id] = (0, -2 * i)
    node_activ_values[node_id] = 0.0
    node_colors[node_id] = token_palette[i % len(token_palette)]
    node_labels[node_id] = processed_tokens[i]

# 2) Hidden layers
layer_count = len(activation_slices)
for layer_idx, layer_data in enumerate(activation_slices, start=1):
    neuron_count = layer_data.shape[1]
    for n_idx in range(neuron_count):
        node_id = f"L{layer_idx}_N{n_idx}"
        G.add_node(node_id)
        positions[node_id] = (layer_idx, -2 * n_idx)

        # Assign a color by category index
        cat_list = list(category_colors.keys())
        cat_label = cat_list[n_idx % len(cat_list)]
        node_colors[node_id] = category_colors[cat_label]

        # Mean activation of all tokens in this neuron
        activation_val = layer_data[:, n_idx].mean()
        node_activ_values[node_id] = float(activation_val)
        node_labels[node_id] = f"{cat_label}\nAct: {activation_val:.2f}"

# 3) Output node
final_node_id = f"L{layer_count + 1}_Out"
G.add_node(final_node_id)
positions[final_node_id] = (layer_count + 1, 0)
node_activ_values[final_node_id] = 0.0
node_colors[final_node_id] = "#FF99CC"
node_labels[final_node_id] = f"Prediction:\n{topic_assigned}"

# Function to measure overlap between two sets of activations
def cooperation_strength(matA, matB, idxA, idxB):
    return float(np.sum(matA[:, idxA] * matB[:, idxB]))

threshold = 0.2
input_threshold = 0.0

# Edges: input -> first hidden
if layer_count > 0:
    first_layer = activation_slices[0]
    for src_i in range(num_tokens):
        for dst_j in range(first_layer.shape[1]):
            strength_val = cooperation_strength(input_embeds, first_layer, src_i, dst_j)
            if strength_val > input_threshold:
                G.add_edge(f"Input_{src_i}", f"L1_N{dst_j}", weight=strength_val)

# Edges: hidden -> hidden
for l_idx in range(layer_count - 1):
    A = activation_slices[l_idx]
    B = activation_slices[l_idx + 1]
    for src_i in range(A.shape[1]):
        for dst_j in range(B.shape[1]):
            strength_val = cooperation_strength(A, B, src_i, dst_j)
            if strength_val > threshold:
                G.add_edge(f"L{l_idx+1}_N{src_i}", f"L{l_idx+2}_N{dst_j}", weight=strength_val)

# Edges: last hidden -> output
if layer_count > 0:
    last_layer = activation_slices[-1]
    for src_i in range(last_layer.shape[1]):
        edge_strength = float(np.sum(last_layer[:, src_i]))
        if abs(edge_strength) > 0.0:
            G.add_edge(f"L{layer_count}_N{src_i}", final_node_id, weight=edge_strength)

# Plotting
fig, ax = plt.subplots(figsize=(18, 12))

# Extract edge data for thickness/color mapping
edges = G.edges(data=True)
edge_vals = [e[2]["weight"] if "weight" in e[2] else 0 for e in edges]
max_val = max(edge_vals) if edge_vals else 1
widths = [2 + (val / max_val) * 8 for val in edge_vals]

nx.draw_networkx_edges(
    G, pos=positions, width=widths, edge_color=edge_vals,
    edge_cmap=plt.cm.Blues, alpha=0.8,
    connectionstyle="arc3,rad=0.2", ax=ax
)

nx.draw_networkx_labels(G, pos=positions, labels=node_labels, font_size=8, ax=ax)

# Highlight nodes with high activation by drawing a large translucent circle behind them
node_values = np.array(list(node_activ_values.values()))
peak_activation = float(node_values.max()) if len(node_values) > 0 else 1.0
highlight_cutoff = 0.7 * peak_activation
glow_size = 2000

for nid in G.nodes():
    val = node_activ_values[nid]
    if val >= highlight_cutoff and peak_activation > 0:
        x_pos, y_pos = positions[nid]
        plt.scatter(x_pos, y_pos, s=glow_size, color="#FFFF99", alpha=0.4, zorder=1)

# Draw the nodes on top
node_order = list(G.nodes())
color_list = [node_colors[n] for n in node_order]
nx.draw_networkx_nodes(G, pos=positions, node_color=color_list, node_size=600, ax=ax)

# Add colorbar for edges
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_val))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax)
cb.set_label("Cooperation Strength")

# Legend
legend_items = []
for lbl, clr in category_colors.items():
    legend_items.append(plt.scatter([], [], color=clr, label=lbl))
legend_items.append(plt.scatter([], [], color="#FF99CC", label="Output Layer"))
ax.legend(handles=legend_items, loc="upper left", fontsize=10)

ax.set_title("Custom GPT-2 Visualization (5 Hidden Layers)")
ax.axis("off")

# Save the figure
save_path = "static/graph.png"
fig.savefig(save_path, format="png", bbox_inches="tight")
print(f"Graph saved to {save_path}")
plt.close(fig)
