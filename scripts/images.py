import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_and_plot(input_file, output_file, pool_file, output_channels):
    # Load the binary files
    input_data = np.fromfile(input_file, dtype=np.float32)
    output_data = np.fromfile(output_file, dtype=np.float32)
    pool_data = np.fromfile(pool_file, dtype=np.float32)

    # Define the shape of the output data
    rows, cols = 28, 28  # Input image size
    out_rows, out_cols = 26, 26  # Output image size after convolution
    pool_rows, pool_cols = 13,13
    output_data = output_data.reshape((output_channels, out_rows, out_cols))
    pool_data = pool_data.reshape((output_channels, pool_rows, pool_cols))
    # print(pool_data)
    print(pool_data.shape)

    # Plot the input image
    plt.figure(figsize=(6, 6))
    plt.imshow(input_data.reshape((rows, cols)), cmap="gray", interpolation="nearest")
    plt.title("Input Image")
    plt.axis("off")
    plt.show()

    # Plot the output channels
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # 4x4 grid of subplots
    axes = axes.flatten()

    for i in range(output_channels):
        ax = axes[i]
        ax.imshow(output_data[i], cmap="gray", interpolation="nearest")
        ax.set_title(f"Channel {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


    # Plot the pooled output channels
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # 4x4 grid again
    axes = axes.flatten()

    for i in range(output_channels):
        ax = axes[i]
        ax.imshow(pool_data[i], cmap="gray", interpolation="nearest")
        ax.set_title(f"Pooled Channel {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Load and plot input/output of convolution"
    )
    parser.add_argument("input_file", type=str, help="Path to the binary input file")
    parser.add_argument("output_file", type=str, help="Path to the binary output file")
    parser.add_argument("pool_file", type=str, help="Path to the binary pool file")
    parser.add_argument(
        "output_channels", type=int, help="Number of output channels to plot"
    )
    args = parser.parse_args()

    # Call function to load and plot the data
    load_and_plot(args.input_file, args.output_file, args.pool_file, args.output_channels)
