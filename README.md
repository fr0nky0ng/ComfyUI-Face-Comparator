# ComfyUI-Face-Comparator
A ComfyUI custom node for detecting the similarity between two faces

# Install

This node is based on insightface, the model of "buffalo_l" will be automatically downloaded the first time you run it.

```bash
cd comfyUI/custom_nodes
git clone https://github.com/fr0nky0ng/ComfyUI-Face-Comparator.git
cd ComfyUI-Face-Comparator
pip install -r requirements.txt
```

# Note
The default cosine similarity threshold is 0.65. The higher the score, the more similar they are. You can adjust this threshold to meet your needs.

<img width="634" height="485" alt="image" src="examples/snapshot.png" />
