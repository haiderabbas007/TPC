import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_architecture(blocks, title, output_file):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlim(0, len(blocks) * 2.2)
    ax.set_ylim(0, 3)
    ax.axis("off")

    x = 0.5
    y = 1.25
    box_w = 1.75
    box_h = 0.75

    for i, block in enumerate(blocks):
        rect = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            linewidth=1.5,
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(rect)

        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            block,
            ha="center",
            va="center",
            fontsize=9,
        )

        if i < len(blocks) - 1:
            arrow = FancyArrowPatch(
                (x + box_w, y + box_h / 2),
                (x + 2.1, y + box_h / 2),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.4,
                color="black",
            )
            ax.add_patch(arrow)

        x += 2.2

    ax.text(
        0.5,
        2.55,
        title,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="center",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {output_file}")


# ============================================================
# Original larger CNN, approximately 17k parameters
# ============================================================

large_cnn_blocks = [
    "Input\n20×256×1",
    "Conv2D\n16 filters\n3×7",
    "MaxPool\n2×2",
    "Dropout\n0.2",
    "Conv2D\n32 filters\n5×15",
    "MaxPool\n2×2",
    "Dropout\n0.2",
    "Flatten",
    "Dense\n128",
    "Dropout\n0.4",
    "Dense\n1 sigmoid",
]

draw_architecture(
    large_cnn_blocks,
    "Original Larger Sub-Hitmap CNN (~17k parameters)",
    "cnn_large_architecture.png",
)


# ============================================================
# Compressed 229-parameter CNN
# ============================================================

small_cnn_blocks = [
    "Input\n20×100×1",
    "Conv2D\n4 filters\n3×5",
    "ReLU",
    "MaxPool\n2×2",
    "Conv2D\n8 filters\n3×7",
    "ReLU",
    "MaxPool\n2×4",
    "Global Avg\nPooling",
    "Dense\n8",
    "ReLU",
    "Dense\n1 logit",
]

draw_architecture(
    small_cnn_blocks,
    "Compressed HLS CNN, 80:180 Crop (229 parameters)",
    "cnn_229_architecture.png",
)