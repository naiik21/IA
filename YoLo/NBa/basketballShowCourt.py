from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import cv2
import supervision as sv


@dataclass
class BasketballCourtShowConfiguration:
    # NBA court dimensions in cm
    length: int = 2880  # 28.65m (94 feet) in cm
    width: int = 1530  # 15.24m (50 feet) in cm
    line_width: int = 1  # [cm] - width of all lines
    
    # Key (paint area) dimensions - NBA has a 16 feet wide key
    key_width: int = 488  # [cm] (16 feet)
    key_length: int = 580  # [cm] (19 feet)
    free_throw_line_distance: int = 580  # [cm] (19 feet) from baseline
    
    # Three-point line dimensions - NBA has 23.75ft (except corners at 22ft)
    three_point_line_radius: int = 724  # [cm] (23.75 feet)
    three_point_corner_distance: int = 669  # [cm] from sideline (creates 22ft in corners)
    three_point_line_straight_distance: int = 669  # [cm] (22 feet) from baseline
    
    # Backboard and rim - NBA standard
    backboard_width: int = 183  # [cm] (6 feet)
    backboard_distance: int = 122  # [cm] (4 feet) from baseline
    rim_diameter: int = 46  # [cm] (18 inches)
    rim_distance: int = 15  # [cm] (6 inches) from backboard

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        half_width = self.width / 2
        half_length = self.length / 2
        
        # Calculate key points
        key_top = half_width + (self.key_width / 2)
        key_bottom = half_width - (self.key_width / 2)
        free_throw_line = self.free_throw_line_distance
        
        # Calculate three-point straight section points
        # These should be exactly at three_point_corner_distance from the center
        three_pt_straight_y_top = half_width + self.three_point_corner_distance
        three_pt_straight_y_bottom = half_width - self.three_point_corner_distance

        
        return [
            # Court outline
            (0, 0),  # 1 - corner
            (0, self.width),  # 2 - corner
            (self.length, self.width),  # 3 - corner
            (self.length, 0),  # 4 - corner
            
            # Center line
            (half_length, 0),  # 5
            (half_length, self.width),  # 6
            
           # Key (paint area) - left side
            (0, key_bottom),  # 7
            (free_throw_line, key_bottom),  # 8
            (free_throw_line, key_top),  # 9
            (0, key_top),  # 10
            
            # Key (paint area) - right side
            (self.length, key_bottom),  # 11
            (self.length - free_throw_line, key_bottom),  # 12
            (self.length - free_throw_line, key_top),  # 13
            (self.length, key_top),  # 14
            
            # Three-point line - left corners
            (0, three_pt_straight_y_bottom),  # 15
            (0, three_pt_straight_y_top),  # 16
            
            # Three-point line - right corners
            (self.length, three_pt_straight_y_bottom),  # 17
            (self.length, three_pt_straight_y_top),  # 18
            
            # Three-point line - left straight section
            (self.three_point_line_straight_distance - 265, three_pt_straight_y_bottom),  # 19
            (self.three_point_line_straight_distance - 265, three_pt_straight_y_top),  # 20
            
            # Three-point line - right straight section
            (self.length - self.three_point_line_straight_distance + 265, three_pt_straight_y_bottom),  # 21
            (self.length - self.three_point_line_straight_distance + 265, three_pt_straight_y_top),  # 22
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Court outline (only sidelines and baselines)
        (1, 2), (2, 3), (3, 4), (4, 1),
        
        # Center line
        (5, 6),
        
        # Key (paint area) - left (only the outer lines)
        (7, 8), (8, 9), (9, 10),
        
        # Key (paint area) - right (only the outer lines)
        (11, 12), (12, 13), (13, 14),
        
        # Three-point line - left
        (15, 19), (20, 16),
        
        # Three-point line - right
        (17, 21), (22, 18),
    ])

    arcs: List[Tuple[int, int, int, int, int]] = field(default_factory=lambda: [
        # Three-point arc - left
        (137, 765, 724, -68, 68),  # NBA 3pt arc (23.75ft)
        
        # Three-point arc - right
        (2743, 765, 724, 112, 248),  # NBA 3pt arc (23.75ft)
    ])


def draw_court_show(
    config: BasketballCourtShowConfiguration,
    background_color: sv.Color = sv.Color(204, 153, 106),  # Dark green
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 10,
    line_thickness: int = 4,
    scale: float = 1
) -> np.ndarray:
    """
    Draws a simplified NBA basketball court with only essential lines.
    
    Args:
        config (BasketballCourtConfiguration): Configuration object.
        background_color (sv.Color): Court background color.
        line_color (sv.Color): Line color.
        padding (int): Padding around the court.
        line_thickness (int): Line thickness.
        scale (float): Scaling factor.
        
    Returns:
        np.ndarray: Image of the simplified basketball court.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    
    court_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw edges
    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=court_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    # Draw arcs
    for center_x, center_y, radius, start_angle, end_angle in config.arcs:
        center = (int(center_x * scale) + padding, int(center_y * scale) + padding)
        scaled_radius = int(radius * scale)
        cv2.ellipse(
            img=court_image,
            center=center,
            axes=(scaled_radius, scaled_radius),
            angle=0,
            startAngle=start_angle,
            endAngle=end_angle,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    return court_image


