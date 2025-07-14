from manim import *
import numpy as np


class AnimatedSphereHeatmap(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        time_tracker = ValueTracker(1.0)

        # Sphere setup
        sphere = Sphere(radius=2, resolution=(50, 50))
        sphere.set_fill(opacity=0.5)  # Base fill opacity
        sphere.set_stroke(width=0.01, color=BLACK)

        # Axes
        axes = ThreeDAxes()

        # Color function dependent on spherical coords and time
        def color_function(point):
            x, y, z = point
            r = np.linalg.norm(point)
            if r == 0:
                phi_val = 0
                theta_val = 0
            else:
                phi_val = np.arccos(z / r)
                theta_val = np.arctan2(y, x)

            t = time_tracker.get_value()
            value = (np.cos(theta_val * t) * np.sin(phi_val * t) + 1) / 2
            value = np.clip(value, 0, 1)
            return interpolate_color(BLUE, RED, value)

        # Dynamic endpoint for moving line
        def get_line_end_coordinates():
            t = time_tracker.get_value()
            return 2 * np.array([
                np.sin(PI / 2 * t) * np.cos(PI / 4 * t),
                np.sin(PI / 2 * t) * np.sin(PI / 4 * t),
                np.cos(PI / 2 * t)
            ])

        # Moving radial line
        moving_line = always_redraw(
            lambda: Line(ORIGIN, get_line_end_coordinates()).set_color(YELLOW)
        )

        # Infinite X/Y/Z lines passing through the moving line's tip
        def axis_line_through_tip(direction):
            return always_redraw(lambda: Line(
                get_line_end_coordinates() - 100 * direction,
                get_line_end_coordinates() + 100 * direction,
                color=GRAY_B,
                stroke_width=1
            ))

        x_axis_line = axis_line_through_tip(np.array([1, 0, 0]))
        y_axis_line = axis_line_through_tip(np.array([0, 1, 0]))
        z_axis_line = axis_line_through_tip(np.array([0, 0, 1]))

        # Heatmap updater
        def update_sphere_colors(mob):
            for submob in mob.submobjects:
                center_point = submob.get_center()
                submob.set_fill(color=color_function(center_point), opacity=0.5)
                submob.set_stroke(width=0.01, color=BLACK)

        sphere.add_updater(update_sphere_colors)

        # Add and animate
        self.add(axes, sphere, moving_line, x_axis_line, y_axis_line, z_axis_line)
        self.play(time_tracker.animate.set_value(6), run_time=6, rate_func=linear)
        self.wait(1)
