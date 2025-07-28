import numpy as np
from shapely.geometry import Point

def calculate_heading(p1, p2):
    """Computes heading in degrees from p1 to p2."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return np.degrees(np.arctan2(dy, dx)) % 360

def calculate_fov(panorama, p_start, p_end):
    """Computes the angle between two vectors (FOV)."""
    vec1 = np.array([p_start.x - panorama.x, p_start.y - panorama.y])
    vec2 = np.array([p_end.x - panorama.x, p_end.y - panorama.y])
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    unit1 = vec1 / norm1
    unit2 = vec2 / norm2
    
    dot = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def get_segment_points(center, radius, start_angle, end_angle, n_points=30):
    """Generates points along a circular arc segment."""
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), n_points)
    return [Point(center.x + radius * np.cos(a), center.y + radius * np.sin(a)) for a in angles]

def split_fov(heading, fov, max_fov=120):
    """Splits a large FOV into multiple smaller sub-FOVs."""
    if fov <= max_fov:
        return [(heading % 360, fov)]
    
    n_parts = int(np.ceil(fov / max_fov))
    sub_fov = fov / n_parts
    
    return [
        ((heading + (i - (n_parts - 1) / 2) * sub_fov) % 360, sub_fov)
        for i in range(n_parts)
    ]


def get_angles(panorama_point: Point, control_point: Point, max_distance: int = 10, max_fov: int = 120):
    """
    Get angles for a given panorama point and control point.
    
    Parameters
    ----------
    panorama_point : Point
        Panorama point
    control_point : Point
        Control point
    max_distance : int
        Maximum distance "seen" from control points
    max_fov : int
        Maximum FOV for segment points
        
    Returns
    -------
    List[Tuple[str, float, float]]
        List of tuples containing direction, heading, and FOV
    """

    output = []
    # Partition angles for N, E, S, W segments
    segment_angles = {
        'N': (45, 135),
        'E': (315, 405),
        'S': (225, 315),
        'W': (135, 225),
    }
    for direction, (start_angle, end_angle) in segment_angles.items():
        pts = get_segment_points(control_point, max_distance, start_angle, end_angle)
        # Get segment endpoints and center
        start_pt, end_pt = pts[0], pts[-1]
        center_idx = len(pts) // 2
        center_pt = pts[center_idx]
        heading_angle = round(calculate_heading(panorama_point, center_pt), 2)
        fov_angle = round(calculate_fov(panorama_point, start_pt, end_pt), 1)
        if fov_angle > max_fov:
            views = split_fov(heading_angle, fov_angle, max_fov)
            for i, (heading, fov) in enumerate(views):
                output.append((direction + str(i+1), heading, fov))
        else:
            output.append((direction, heading_angle, fov_angle))
    return output
