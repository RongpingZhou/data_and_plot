import cv2
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

img = np.load("original_file.npy")   # shape (H, W, 3)
print(img.shape)

plt.imshow(img[:, :, [2, 1, 0]].astype(np.uint8))
plt.axis("off")
plt.show()

img = np.load("cropped_file.npy")   # shape (H, W, 3)
print(img.shape)

plt.imshow(img[:, :, [2, 1, 0]].astype(np.uint8))
plt.axis("off")
plt.show()

img1 = np.load("preprocessed_file.npy")   # shape (H, W, 3)
print(img.shape)

plt.imshow(img1.astype(np.uint8))
plt.axis("off")
plt.show()

plt.imshow(img1.astype(np.uint8), cmap="gray")
plt.axis("off")
plt.show()

img = np.load("env_obs_file.npy")   # shape (H, W, 3)
print(img.shape)

plt.imshow(img.astype(np.uint8))
plt.axis("off")
plt.show()

img = np.load("game_screen_file.npy")   # shape (H, W, 3)
print(img.shape)

plt.imshow(img.astype(np.uint8))
plt.axis("off")
plt.show()

img2 = np.load("obs_resized_file.npy")   # shape (H, W, 3)
print(img2.shape)

plt.imshow(img2.astype(np.uint8))
plt.axis("off")
plt.show()

plt.imshow(img2.astype(np.uint8), cmap="gray")
plt.axis("off")
plt.show()

def image_diff(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Resize to match
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Absolute pixel difference
    diff = cv2.absdiff(img1, img2)

    # Convert to grayscale for metrics + visualization
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Percentage difference
    diff_percent = diff_gray.sum() / (255 * diff_gray.size) * 100

    return diff, diff_gray, diff_percent

# diff, diff_gray, diff_percent = image_diff("a.jpg", "obs_resized_file")
# print("Difference %:", diff_percent)

def compute_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    # Ensure same shape
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    return diff

def plot_heatmap(diff):
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap="hot")
    plt.colorbar(label="Difference Intensity")
    plt.title("Difference Heatmap")
    plt.axis("off")
    plt.show()
    
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(diff):
    h, w = diff.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, diff, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_title("3D Surface of Array Difference")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Difference")
    plt.show()

def plot_3d_surface_no_color(diff):
    h, w = diff.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, diff,
        cmap=None,
        color='lightgray',     # uniform color
        edgecolor='none'       # remove grid lines
    )

    ax.set_title("3D Surface of Array Difference (No Colors)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Difference")

    plt.show()

def highlight_diff_single_color(a, b, color="red"):
    a = np.asarray(a)
    b = np.asarray(b)

    # Absolute difference
    diff = np.abs(a - b)

    # Binary mask: 1 where different, 0 where same
    mask = (diff > 0).astype(np.uint8)

    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap=plt.cm.colors.ListedColormap(["black", color]))
    plt.title("Difference Highlighted (Single Color)")
    plt.axis("off")
    plt.show()
    
def diff_percent(diff):
    diff = diff.astype(np.float32)
    return diff.sum() / (diff.size * diff.max()) * 100

def pixel_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.abs(a.astype(np.float32) - b.astype(np.float32))

def show_single_color_heatmap(diff, color="red", threshold=0):
    mask = (diff > threshold).astype(np.uint8)
    cmap = ListedColormap(["black", color])

    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap=cmap)
    plt.title(f"Pixel Difference (color={color}, threshold={threshold})")
    plt.axis("off")
    plt.show()
    
def diff_percent(diff):
    return diff.sum() / (diff.size * 255.0) * 100

def percent_pixels_different(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Pixel-by-pixel comparison
    diff_mask = a != b

    # For RGB images, collapse channel differences
    if diff_mask.ndim == 3:
        diff_mask = diff_mask.any(axis=2)

    # Percentage of pixels that differ
    return diff_mask.mean() * 100

# def plot_3d_surface(diff):
#     h, w = diff.shape
#     X, Y = np.meshgrid(np.arange(w), np.arange(h))

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot_surface(
#         X, Y, diff,
#         cmap=None,
#         color='lightgray',   # uniform color
#         edgecolor='none'     # smooth surface
#     )

#     ax.set_title("3D Surface of Pixel Difference")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Difference")

#     plt.show()
        
diff = pixel_diff(img1, img2)

plot_3d_surface(diff)

diff_flipped = np.flipud(diff)

plot_3d_surface(diff_flipped)

diff_flipped = np.fliplr(diff)
plot_3d_surface(diff_flipped)


percent = percent_pixels_different(img1, img2)
print("Percentage of pixels different:", percent)

percent = diff_percent(diff)
print("Difference %:", percent)
show_single_color_heatmap(diff, color="red", threshold=0)
# show_single_color_heatmap(diff, color="red", threshold=5)
# show_single_color_heatmap(diff, color="red", threshold=10)
# show_single_color_heatmap(diff, color="red", threshold=15)
# show_single_color_heatmap(diff, color="red", threshold=20)
        
diff = compute_diff(img1, img2)
# highlight_diff_single_color(img1, img2)
print("Difference %:", diff_percent(diff))
plot_heatmap(diff)
plot_3d_surface(diff)
plot_3d_surface_no_color(diff)