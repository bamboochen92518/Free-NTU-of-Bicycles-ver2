from PIL import Image
import numpy as np

def render_point_cloud(points, colors, img_shape, intrinsics):
    height, width = 294, 518
    fx, fy, cx, cy = intrinsics
    rendered_img = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    valid_mask = points[:, 2] > 0
    points = points[valid_mask]
    colors = colors[valid_mask]
    if len(points) == 0:
        return Image.fromarray(rendered_img)
    u = (fx * points[:, 0] / points[:, 2] + cx).astype(np.int32)
    v = (fy * points[:, 1] / points[:, 2] + cy).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid], v[valid]
    z = points[valid, 2]
    colors = colors[valid]
    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            rendered_img[v[i], u[i]] = colors[i]
    rendered_img = Image.fromarray(rendered_img)
    rendered_img = rendered_img.resize((img_shape[1], img_shape[0]), Image.LANCZOS)
    return rendered_img