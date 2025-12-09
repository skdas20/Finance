import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, FancyBboxPatch

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # IEEE Monochromatic Style: Black, White, Grays
    # Components
    boxes = {
        "Market Data": (0.05, 0.4, 0.15, 0.2),
        "LSTM (Forecaster)": (0.3, 0.6, 0.2, 0.2),
        "Feature Eng.": (0.3, 0.2, 0.2, 0.2),
        "PPO Agent (Actor)": (0.6, 0.4, 0.15, 0.2),
        "Action (Buy/Sell)": (0.85, 0.4, 0.1, 0.2)
    }
    
    # Draw Boxes
    for name, (x, y, w, h) in boxes.items():
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", 
                              ec="black", fc="white", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=10, weight='bold')

    # Arrows
    # Data -> LSTM
    ax.annotate("", xy=(0.3, 0.7), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle="->", color="black"))
    # Data -> Feat Eng
    ax.annotate("", xy=(0.3, 0.3), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle="->", color="black"))
    
    # LSTM -> PPO
    ax.annotate("Forecast", xy=(0.6, 0.55), xytext=(0.5, 0.7), arrowprops=dict(arrowstyle="->", color="black"))
    # Feat -> PPO
    ax.annotate("Indicators", xy=(0.6, 0.45), xytext=(0.5, 0.3), arrowprops=dict(arrowstyle="->", color="black"))
    
    # PPO -> Action
    ax.annotate("Decision", xy=(0.85, 0.5), xytext=(0.75, 0.5), arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title("Fig. 1. Hybrid LSTM-PPO Architecture for Autonomous Trading", y=-0.1, fontsize=11)
    plt.tight_layout()
    plt.savefig("architecture_diagram.png", dpi=300, bbox_inches='tight')
    print("Architecture diagram saved.")

if __name__ == "__main__":
    draw_architecture_diagram()
