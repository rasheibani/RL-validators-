import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Function to draw a box
def draw_box(ax, x, y, width, height, text, color='lightblue'):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", linewidth=1.5, edgecolor='black', facecolor=color)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))

# Draw input layer
draw_box(ax, 0, 4, 2, 1, 'Input Layer\n(State Representation)', color='lightgreen')

# Draw feature extractor layers
draw_box(ax, 3, 6, 2.5, 1, 'Feature Extractor Layer 1\n(Fully Connected + ReLU)', color='lightblue')
draw_box(ax, 3, 4, 2.5, 1, 'Feature Extractor Layer 2\n(Fully Connected + ReLU)', color='lightblue')

# Draw actor-critic policy network layers
draw_box(ax, 6.5, 6, 2.5, 1, 'Policy Network\n(Fully Connected + ReLU)', color='lightyellow')
draw_box(ax, 6.5, 4, 2.5, 1, 'Value Network\n(Fully Connected)', color='lightyellow')

# Draw output layer
draw_box(ax, 10, 5, 2, 1, 'Output Layer\n(Action Probabilities & Value)', color='salmon')

# Draw arrows to represent connections
ax.annotate('', xy=(2, 4.5), xytext=(3, 6.5), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(2, 4.5), xytext=(3, 4.5), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(5.5, 6.5), xytext=(6.5, 6.5), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(5.5, 4.5), xytext=(6.5, 4.5), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(9, 6), xytext=(10, 5.5), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(9, 4.5), xytext=(10, 5.5), arrowprops=dict(facecolor='black', arrowstyle='->'))

# Set axis limits and remove axes for better visualization
ax.set_xlim(-1, 13)
ax.set_ylim(0, 9)
ax.axis('off')

# Display the architecture diagram
plt.title('Architecture of Custom Deep Q-Network (DQN) with Feature Extractor and Policy Network')
plt.show()
