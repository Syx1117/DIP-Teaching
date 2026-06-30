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
    if image is None:
        return None

    x, y = int(evt.index[0]), int(evt.index[1])

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 4, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 4, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image,
            tuple(points_src[i]),
            tuple(points_dst[i]),
            (0, 255, 0),
            2,
            tipLength=0.25,
        )

    return marked_image


def _mls_affine_map(height, width, control_src, control_dst, alpha=1.0, eps=1e-8):
    """Compute affine Moving Least Squares mapping for every image pixel.

    The returned map sends coordinates from control_src to control_dst. The
    implementation processes pixels in blocks to keep memory usage bounded.
    """
    p = np.asarray(control_src, dtype=np.float64)
    q = np.asarray(control_dst, dtype=np.float64)

    total_pixels = height * width
    mapped = np.empty((total_pixels, 2), dtype=np.float64)
    block_size = 65536

    for start in range(0, total_pixels, block_size):
        end = min(start + block_size, total_pixels)
        flat = np.arange(start, end, dtype=np.int64)

        v = np.empty((end - start, 2), dtype=np.float64)
        v[:, 0] = flat % width
        v[:, 1] = flat // width

        diff = p[None, :, :] - v[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest = np.argmin(dist2, axis=1)
        nearest_dist2 = dist2[np.arange(end - start), nearest]

        weights = 1.0 / (np.maximum(dist2, eps) ** alpha)
        weight_sum = np.sum(weights, axis=1, keepdims=True)

        p_star = (weights @ p) / weight_sum
        q_star = (weights @ q) / weight_sum
        phat = p[None, :, :] - p_star[:, None, :]
        qhat = q[None, :, :] - q_star[:, None, :]

        wx = weights * phat[:, :, 0]
        wy = weights * phat[:, :, 1]

        a00 = np.sum(wx * phat[:, :, 0], axis=1)
        a01 = np.sum(wx * phat[:, :, 1], axis=1)
        a11 = np.sum(wy * phat[:, :, 1], axis=1)

        b00 = np.sum(wx * qhat[:, :, 0], axis=1)
        b01 = np.sum(wx * qhat[:, :, 1], axis=1)
        b10 = np.sum(wy * qhat[:, :, 0], axis=1)
        b11 = np.sum(wy * qhat[:, :, 1], axis=1)

        delta = v - p_star
        out = v + (q_star - p_star)

        det = a00 * a11 - a01 * a01
        det_threshold = eps * np.maximum((a00 + a11) ** 2, 1.0)
        affine_ok = np.abs(det) > det_threshold

        if np.any(affine_ok):
            d = det[affine_ok]
            m00 = (a11[affine_ok] * b00[affine_ok] - a01[affine_ok] * b10[affine_ok]) / d
            m01 = (a11[affine_ok] * b01[affine_ok] - a01[affine_ok] * b11[affine_ok]) / d
            m10 = (-a01[affine_ok] * b00[affine_ok] + a00[affine_ok] * b10[affine_ok]) / d
            m11 = (-a01[affine_ok] * b01[affine_ok] + a00[affine_ok] * b11[affine_ok]) / d

            dx = delta[affine_ok, 0]
            dy = delta[affine_ok, 1]
            out[affine_ok, 0] = dx * m00 + dy * m10 + q_star[affine_ok, 0]
            out[affine_ok, 1] = dx * m01 + dy * m11 + q_star[affine_ok, 1]

        # Degenerate control layouts cannot define a full affine transform.
        # Fall back to similarity MLS, then to local weighted translation.
        similarity_needed = ~affine_ok
        if np.any(similarity_needed):
            mu = np.sum(
                weights * (phat[:, :, 0] ** 2 + phat[:, :, 1] ** 2),
                axis=1,
            )
            similarity_ok = similarity_needed & (mu > eps)

            if np.any(similarity_ok):
                a = np.sum(
                    weights
                    * (phat[:, :, 0] * qhat[:, :, 0] + phat[:, :, 1] * qhat[:, :, 1]),
                    axis=1,
                ) / np.maximum(mu, eps)
                b = np.sum(
                    weights
                    * (phat[:, :, 0] * qhat[:, :, 1] - phat[:, :, 1] * qhat[:, :, 0]),
                    axis=1,
                ) / np.maximum(mu, eps)

                dx = delta[similarity_ok, 0]
                dy = delta[similarity_ok, 1]
                out[similarity_ok, 0] = (
                    dx * a[similarity_ok]
                    - dy * b[similarity_ok]
                    + q_star[similarity_ok, 0]
                )
                out[similarity_ok, 1] = (
                    dx * b[similarity_ok]
                    + dy * a[similarity_ok]
                    + q_star[similarity_ok, 1]
                )

        exact = nearest_dist2 < eps
        if np.any(exact):
            out[exact] = q[nearest[exact]]

        mapped[start:end] = out

    return mapped.reshape(height, width, 2)


# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    if image is None:
        return None

    warped_image = np.asarray(image).copy()
    source_pts = np.asarray(source_pts, dtype=np.float64).reshape(-1, 2)
    target_pts = np.asarray(target_pts, dtype=np.float64).reshape(-1, 2)

    pair_count = min(len(source_pts), len(target_pts))
    if pair_count == 0:
        return warped_image

    source_pts = source_pts[:pair_count]
    target_pts = target_pts[:pair_count]
    height, width = warped_image.shape[:2]

    # Inverse warping: each output pixel is mapped back to the source image.
    coord_map = _mls_affine_map(
        height,
        width,
        target_pts,
        source_pts,
        alpha=float(alpha),
        eps=float(eps),
    )
    map_x = coord_map[:, :, 0].astype(np.float32)
    map_y = coord_map[:, :, 1].astype(np.float32)

    return cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


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


def build_demo():
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

    return demo


if __name__ == "__main__":
    build_demo().launch()
