import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    RBF-based image deformation
    """

    h, w = image.shape[:2]

    if len(source_pts) == 0:
        return image

    source_pts = np.asarray(source_pts, dtype=np.float32)
    target_pts = np.asarray(target_pts, dtype=np.float32)

    n = len(source_pts)

    # ---------------------------------
    # RBF interpolation
    # ---------------------------------

    def phi(r):
        return np.sqrt(r * r + eps)

    # Compute interpolation matrix
    K = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(source_pts[i] - source_pts[j])
            K[i, j] = phi(r)

    displacement = target_pts - source_pts

    wx = np.linalg.solve(K + eps * np.eye(n), displacement[:, 0])
    wy = np.linalg.solve(K + eps * np.eye(n), displacement[:, 1])

    # ---------------------------------
    # Backward warping
    # ---------------------------------

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    map_x = xx.astype(np.float32)
    map_y = yy.astype(np.float32)

    dx = np.zeros_like(map_x)
    dy = np.zeros_like(map_y)

    for i in range(n):

        r = np.sqrt(
            (map_x - source_pts[i, 0]) ** 2 +
            (map_y - source_pts[i, 1]) ** 2 +
            eps
        )

        basis = phi(r)

        dx += wx[i] * basis
        dy += wy[i] * basis

    map_x = map_x - dx
    map_y = map_y - dy

    warped_image = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()

