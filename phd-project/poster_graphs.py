from manim import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from laser_phase_plate import generate_wavefronts_start_to_end_gaussian, X_R, W_0, COLOR_UNPERTURBED_AMPLITUDE

BACKGROUND_COLOR = "#29505B"
TEXT_COLOR = "#D8E1E3"
BACKGROUND_NOISE_PATH = "phd-project/hue_noise_background.png"
SAMPLE_COLOR = BLUE
SVG_PATH = r"phd-project/sea horse.svg"
SATURATION = 0.7
BRIGHTNESS = 0.7
import cv2

class DualImageScene(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        x_len = 5
        # Axes
        left_axes = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=x_len,
            y_length=x_len,
            axis_config={"include_tip": False},
            tips=False
        ).move_to(LEFT * x_len/2+1*DOWN)
        right_axes = left_axes.copy().next_to(left_axes, RIGHT, buff=1.5)

        # Titles
        left_title = Tex("Intensity", color=TEXT_COLOR).next_to(left_axes, UP).scale(0.9)
        right_title = Tex("Phase", color=TEXT_COLOR).next_to(right_axes, UP).scale(0.9)
        sup_title = Tex("Electron's wave function at the camera", color=TEXT_COLOR).to_corner(UP).scale(1.1)

        # Create and save grayscale noise
        intensity_noise = np.random.normal(loc=0.5, scale=0.05, size=(100, 100))
        intensity_noise = np.clip(intensity_noise, 0, 1)
        plt.imsave("phd-project/noise_img_raw.png", intensity_noise, cmap="gray")

        # Apply blur using OpenCV
        img_gray = cv2.imread("phd-project/noise_img_raw.png", cv2.IMREAD_GRAYSCALE)
        img_gray_blurred = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=1.5)
        cv2.imwrite("phd-project/noise_img_blur.png", img_gray_blurred)

        # Load as Manim object
        intensity_image = ImageMobject("phd-project/noise_img_blur.png")
        intensity_image.scale_to_fit_width(left_axes.width)
        intensity_image.move_to(left_axes.c2p(0, 0))
        intensity_image.set_z_index(-1)

        # Generate hue-based image
        h = np.clip(np.random.normal(loc=0.076, scale=0.1, size=(100, 100)), 0, 1)
        s = np.full_like(h, SATURATION) / 1.7
        v = np.full_like(h, BRIGHTNESS)

        hsv_pixels = np.stack([h, s, v], axis=-1)
        rgb_pixels = np.zeros_like(hsv_pixels)
        for i in range(100):
            for j in range(100):
                rgb_pixels[i, j] = colorsys.hsv_to_rgb(*hsv_pixels[i, j])

        plt.imsave("phd-project/hue_noise_raw.png", rgb_pixels)

        # Read and blur with OpenCV
        img_rgb = cv2.imread("phd-project/hue_noise_raw.png")
        img_rgb_blurred = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=1.5)
        cv2.imwrite(BACKGROUND_NOISE_PATH, img_rgb_blurred)

        # Load as Manim object
        hue_noise_image = ImageMobject(BACKGROUND_NOISE_PATH)
        hue_noise_image.scale_to_fit_width(right_axes.width)
        hue_noise_image.move_to(right_axes.c2p(0, 0))
        hue_noise_image.set_z_index(-2)

        # SVG shape
        phase_image = SVGMobject(SVG_PATH)
        phase_image.set_fill(SAMPLE_COLOR, opacity=0.5)
        phase_image.set_stroke(width=0)
        phase_image.scale_to_fit_width(right_axes.width).scale(0.3)
        phase_image.move_to(right_axes.c2p(0, 0))
        phase_image.set_z_index(-1)

        # Combine
        self.add(intensity_image, hue_noise_image, phase_image)
        self.add(left_axes, right_axes, left_title, right_title, sup_title)




COLOR_ARC = BLUE
COLOR_CIRCLE = GRAY
COLOR_LINE = GOLD_B
COLOR_ARROW = TEAL
WIDTH_LINES_THICK = 3.5
WIDTH_LINES_THIN = 1.5


class PhaseShift(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        radius = 3
        arc_angle = PI / 6

        middle_point = (0, -1.5, 0)

        # Style samples
        legend_signal = Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_ARC, stroke_width=WIDTH_LINES_THICK)
        legend_dc = Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_LINE, stroke_width=WIDTH_LINES_THICK)
        legend_total = Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_ARROW, stroke_width=WIDTH_LINES_THICK)

        # Positioning
        legend_signal.next_to(middle_point, LEFT)
        legend_dc.next_to(legend_signal, 3 * UP)
        legend_total.next_to(legend_signal, 3 * DOWN)

        # Labels
        label_dc = Tex("DC", font_size=35, color=TEXT_COLOR).next_to(legend_dc, RIGHT)
        label_signal = Tex("Signal", font_size=35, color=TEXT_COLOR).next_to(legend_signal, RIGHT)
        label_total = Tex("Total amplitude", font_size=35, color=TEXT_COLOR).next_to(legend_total, RIGHT)

        # Axes setup
        left_axes = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False},
            tips=False, color=TEXT_COLOR
        ).shift(4.5*LEFT+1.5*DOWN)

        right_axes = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False},
            tips=False, color=TEXT_COLOR
        ).shift(4.5*RIGHT+1.5*DOWN)

        x_label_left = left_axes.get_x_axis_label(
            MathTex(r"\Re\left(\psi\right)", color=TEXT_COLOR).scale(0.6)
        )
        y_label_left = left_axes.get_y_axis_label(
            MathTex(r"\Im\left(\psi\right)", color=TEXT_COLOR).scale(0.6),
            edge=UP,
            direction=2*UP,
        )

        x_label_right = right_axes.get_x_axis_label(
            MathTex(r"\Re\left(\psi\right)", color=TEXT_COLOR).scale(0.6)
        )
        y_label_right = right_axes.get_y_axis_label(
            MathTex(r"\Im\left(\psi\right)", color=TEXT_COLOR).scale(0.6),
            edge=UP,
            direction=2*UP,
        )
        # Titles
        left_title = Tex("With original DC", color=TEXT_COLOR).next_to(left_axes, 2.7*UP).scale(0.9)
        right_title = Tex("With phase shifted DC", color=TEXT_COLOR).next_to(right_axes, 2.7*UP).scale(0.9)
        sup_title = Tex(r"Wave function at some specific pixel: $\psi\left(x_{0},y_{0}\right)$", color=TEXT_COLOR).shift(3.2 * UP).scale(1.2)

        # Full circle (parametric) on both sides
        circle_func = lambda t: radius * np.array([np.cos(t), np.sin(t), 0])
        left_circle = left_axes.plot_parametric_curve(
            circle_func,
            use_vectorized=False,
            t_range=[0, TAU],
            color=COLOR_CIRCLE,
            stroke_width=WIDTH_LINES_THIN
        ).move_to(left_axes.c2p(0, 0))

        right_circle = right_axes.plot_parametric_curve(
            circle_func,
            use_vectorized=False,
            t_range=[0, TAU],
            color=COLOR_CIRCLE,
            stroke_width=WIDTH_LINES_THIN
        )


        # Left arc (on circle)
        left_arc = left_axes.plot_parametric_curve(
            circle_func,
            use_vectorized=False,
            t_range=[0, arc_angle],
            color=COLOR_ARC,
            stroke_width=WIDTH_LINES_THICK
        )

        # Right arc (lifted vertically)
        lifted_func = lambda t: circle_func(t) + np.array([-radius, radius, 0])
        right_arc = right_axes.plot_parametric_curve(
            lifted_func,
            use_vectorized=False,
            t_range=[0, arc_angle],
            color=COLOR_ARC,
            stroke_width=WIDTH_LINES_THICK
        )

        # Arrows
        left_end = circle_func(arc_angle)
        line_left = Line(
            start=left_axes.c2p(0, 0),
            end=left_axes.c2p(radius, 0),
            color=COLOR_LINE,
            stroke_width=WIDTH_LINES_THICK
        )
        arrow_left = Arrow(
            start=left_axes.c2p(0, 0),
            end=left_axes.c2p(*left_end[:2]),
            buff=0,
            color=COLOR_ARROW,
            stroke_width=WIDTH_LINES_THICK,
            max_tip_length_to_length_ratio=0.15
        )

        right_end = lifted_func(arc_angle)
        line_right = Line(
            start=right_axes.c2p(0, 0),
            end=right_axes.c2p(0, radius),
            color=COLOR_LINE,
            stroke_width=WIDTH_LINES_THICK
        )
        arrow_right = Arrow(
            start=right_axes.c2p(0, 0),
            end=right_axes.c2p(*right_end[:2]),
            buff=0,
            color=COLOR_ARROW,
            stroke_width=WIDTH_LINES_THICK,
            max_tip_length_to_length_ratio=0.1
        )


        # Display all
        self.add(
            left_axes, right_axes, x_label_left, y_label_left, x_label_right, y_label_right,
            left_title, right_title, sup_title,
            left_circle, right_circle,
            left_arc, right_arc,
            line_left, line_right,
            arrow_left, arrow_right,
            legend_signal, legend_dc, legend_total,
            label_signal, label_dc, label_total
        )


class PhaseShiftAndAttenuation(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        radius = 3
        arc_angle = PI / 6
        attenuation = 1 / 5

        middle_point = (0, -1.5, 0)

        # Style samples
        legend_signal = Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_ARC, stroke_width=WIDTH_LINES_THICK)
        legend_dc = Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_LINE, stroke_width=WIDTH_LINES_THICK)
        legend_total = Line(LEFT * 0.3, RIGHT * 0.3, color=COLOR_ARROW, stroke_width=WIDTH_LINES_THICK)

        legend_signal.next_to(middle_point, LEFT)
        legend_dc.next_to(legend_signal, 3 * UP)
        legend_total.next_to(legend_signal, 3 * DOWN)

        label_dc = Tex("DC", font_size=35, color=TEXT_COLOR).next_to(legend_dc, RIGHT)
        label_signal = Tex("Signal", font_size=35, color=TEXT_COLOR).next_to(legend_signal, RIGHT)
        label_total = Tex("Total amplitude", font_size=35, color=TEXT_COLOR).next_to(legend_total, RIGHT)

        left_axes = Axes(
            x_range=[-4, 4], y_range=[-4, 4], x_length=4, y_length=4,
            axis_config={"include_tip": False}, tips=False, color=TEXT_COLOR
        ).shift(4.5 * LEFT + 1.5 * DOWN)

        right_axes = Axes(
            x_range=[-4, 4], y_range=[-4, 4], x_length=4, y_length=4,
            axis_config={"include_tip": False}, tips=False, color=TEXT_COLOR
        ).shift(4.5 * RIGHT + 1.5 * DOWN)

        x_label_left = left_axes.get_x_axis_label(MathTex(r"\Re\left(\psi\right)", color=TEXT_COLOR).scale(0.6))
        y_label_left = left_axes.get_y_axis_label(MathTex(r"\Im\left(\psi\right)", color=TEXT_COLOR).scale(0.6), edge=UP, direction=2 * UP)
        x_label_right = right_axes.get_x_axis_label(MathTex(r"\Re\left(\psi\right)", color=TEXT_COLOR).scale(0.6))
        y_label_right = right_axes.get_y_axis_label(MathTex(r"\Im\left(\psi\right)", color=TEXT_COLOR).scale(0.6), edge=UP, direction=2 * UP)

        left_title = Tex("With original DC", color=TEXT_COLOR).next_to(left_axes, 2.7 * UP).scale(0.9)
        right_title = Tex("With attenuated and phase shifted DC", color=TEXT_COLOR).next_to(right_axes, 2.7 * UP).scale(0.9)
        sup_title = Tex(r"Wave function at some specific pixel: $\psi\left(x_{0},y_{0}\right)$", color=TEXT_COLOR).shift(3.2 * UP).scale(1.2)

        circle_func = lambda t: radius * np.array([np.cos(t), np.sin(t), 0])
        left_circle = left_axes.plot_parametric_curve(circle_func, use_vectorized=False, t_range=[0, TAU], color=COLOR_CIRCLE, stroke_width=WIDTH_LINES_THIN).move_to(left_axes.c2p(0, 0))
        right_circle = right_axes.plot_parametric_curve(circle_func, use_vectorized=False, t_range=[0, TAU], color=COLOR_CIRCLE, stroke_width=WIDTH_LINES_THIN)

        left_arc = left_axes.plot_parametric_curve(circle_func, use_vectorized=False, t_range=[0, arc_angle], color=COLOR_ARC, stroke_width=WIDTH_LINES_THICK)

        # Right arc lifts by attenuated DC length instead of full radius
        lifted_func = lambda t: circle_func(t) + np.array([-radius, radius * attenuation, 0])
        right_arc = right_axes.plot_parametric_curve(lifted_func, use_vectorized=False, t_range=[0, arc_angle], color=COLOR_ARC, stroke_width=WIDTH_LINES_THICK)

        left_end = circle_func(arc_angle)
        line_left = Line(start=left_axes.c2p(0, 0), end=left_axes.c2p(radius, 0), color=COLOR_LINE, stroke_width=WIDTH_LINES_THICK)
        arrow_left = Arrow(start=left_axes.c2p(0, 0), end=left_axes.c2p(*left_end[:2]), buff=0, color=COLOR_ARROW, stroke_width=WIDTH_LINES_THICK, max_tip_length_to_length_ratio=0.15)

        right_end = lifted_func(arc_angle)
        line_right = Line(start=right_axes.c2p(0, 0), end=right_axes.c2p(0, radius * attenuation), color=COLOR_LINE, stroke_width=WIDTH_LINES_THICK)
        arrow_right = Arrow(start=right_axes.c2p(0, 0), end=right_axes.c2p(*right_end[:2]), buff=0, color=COLOR_ARROW, stroke_width=WIDTH_LINES_THICK, max_tip_length_to_length_ratio=0.1)

        self.add(
            left_axes, right_axes, x_label_left, y_label_left, x_label_right, y_label_right,
            left_title, right_title, sup_title,
            left_circle, right_circle,
            left_arc, right_arc,
            line_left, line_right,
            arrow_left, arrow_right,
            legend_signal, legend_dc, legend_total,
            label_signal, label_dc, label_total
        )


COLOR_LENS = BLUE
COLOR_MIRROR = GREEN
COLOR_LASER = RED
COLOR_ELECTRON = PURPLE

class Cavity(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        mm = 0.1
        global_shift = -6
        global_vertical_shift = 0

        lens_R_right = 5.49 * mm
        lens_R_left = 24.21 * mm
        lens_D = 7.75 * mm
        lens_thickness = 2.91 * mm
        lens_left_location = 5 * mm
        lens_right_location = lens_left_location + lens_thickness
        lens_arc_angle_left = np.arcsin(lens_D / (2 * lens_R_left))
        lens_arc_angle_right = np.arcsin(lens_D / (2 * lens_R_right))

        lens_arc_left = Arc(
            radius=lens_R_left,
            start_angle=PI - lens_arc_angle_left,
            angle=2 * lens_arc_angle_left,
            color=COLOR_LENS,
            stroke_width=WIDTH_LINES_THICK
        ).move_to([lens_left_location + global_shift, global_vertical_shift, 0])

        lens_arc_right = Arc(
            radius=lens_R_right,
            start_angle=-lens_arc_angle_right,
            angle=2 * lens_arc_angle_right,
            color=COLOR_LENS,
            stroke_width=WIDTH_LINES_THICK
        ).move_to([lens_right_location + global_shift, global_vertical_shift, 0])

        small_mirror_R = 5 * mm
        small_mirror_D = 7.75 * mm
        small_mirror_location = -5 * mm
        small_mirror_arc_angle = np.arcsin(small_mirror_D / (2 * small_mirror_R))

        small_mirror_arc = Arc(
            radius=small_mirror_R,
            start_angle=np.pi - small_mirror_arc_angle,
            angle=2 * small_mirror_arc_angle,
            color=COLOR_MIRROR,
            stroke_width=WIDTH_LINES_THICK
        ).move_to([small_mirror_location + global_shift, global_vertical_shift, 0])

        big_mirror_R = 200*mm
        big_mirror_location = 81*mm
        big_mirror_D = 25.4*mm
        big_mirror_arc_angle = np.arcsin(big_mirror_D / (2 * big_mirror_R))

        big_mirror_arc = Arc(
            radius=big_mirror_R,
            start_angle=-big_mirror_arc_angle,
            angle=2 * big_mirror_arc_angle,
            color=COLOR_MIRROR,
            stroke_width=WIDTH_LINES_THICK
        ).move_to([big_mirror_location + global_shift, global_vertical_shift, 0])
        a = ValueTracker(0.4)
        b = ValueTracker(0.6)
        c = ValueTracker(0.5)
        WAVELENGTH = 0.2
        waves_left = generate_wavefronts_start_to_end_gaussian(start_point=[small_mirror_location + global_shift+1*mm, global_vertical_shift, 0],
                                                          end_point=[lens_left_location + global_shift-1*mm, global_vertical_shift, 0],
                                                          tracker=a,
                                                          wavelength=WAVELENGTH,
                                                          x_R=X_R/20,
                                                          w_0=W_0/5,
                                                          center=[(small_mirror_location + lens_left_location) / 2 + global_shift, global_vertical_shift, 0],
                                                          colors_generator=lambda t: COLOR_LASER)
        waves_right = generate_wavefronts_start_to_end_gaussian(start_point=[lens_right_location + global_shift+1*mm, global_vertical_shift, 0],
                                                               end_point=[big_mirror_location + global_shift+1*mm, global_vertical_shift, 0],
                                                               tracker=b,
                                                               wavelength=WAVELENGTH,
                                                               x_R=X_R*2,
                                                               w_0=W_0/1.5,
                                                               center=None,
                                                               colors_generator=lambda t: COLOR_LASER)
        waves_lens = generate_wavefronts_start_to_end_gaussian(start_point=[lens_left_location + global_shift, global_vertical_shift, 0],
                                                                end_point=[lens_right_location + global_shift, global_vertical_shift, 0],
                                                                tracker=c,
                                                                wavelength=WAVELENGTH,
                                                                x_R=X_R,
                                                                w_0=W_0,
                                                                center=[small_mirror_location + global_shift, global_vertical_shift, 0],
                                                                colors_generator=lambda t: COLOR_LASER)

        electron_line = DashedLine(start=[global_shift, 1 + global_vertical_shift, 0], end=[global_shift, -1 + global_vertical_shift, 0],
                             color=COLOR_UNPERTURBED_AMPLITUDE, stroke_width=WIDTH_LINES_THICK)

        self.add(
            lens_arc_left, lens_arc_right,
            small_mirror_arc, big_mirror_arc, waves_left, waves_right, waves_lens, electron_line
        )

        # Legend items setup
        legend_items = [
            {"label": "Cavity mirrors", "color": COLOR_MIRROR, "type": Line},
            {"label": "Lens", "color": COLOR_LENS, "type": Line},
            {"label": "Laser beam", "color": COLOR_LASER, "type": Line},
            {"label": "Electron beam", "color": COLOR_UNPERTURBED_AMPLITUDE, "type": DashedLine}
        ]

        legend_lines = []
        legend_labels = []

        legend_start = [3.5, 0.75, 0]  # Top-right relative position

        for i, item in enumerate(legend_items):
            line = item["type"](
                LEFT * 0.4, RIGHT * 0.4,
                color=item["color"],
                stroke_width=WIDTH_LINES_THICK
            ).move_to([legend_start[0], legend_start[1] - i * 0.5, 0])

            label = Tex(item["label"], font_size=40, color=TEXT_COLOR).next_to(line, RIGHT, buff=0.3)

            legend_lines.append(line)
            legend_labels.append(label)


        # Add legend to scene
        self.add(*legend_lines, *legend_labels)


from manim import config as global_config
config["frame_width"] = 18
# config = global_config.copy()

class TiltToleranceGraph(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        font_size = 66
        # Generate dummy data
        NA = np.logspace(np.log10(0.03), np.log10(0.2), 5)
        colors = [GOLD_B, BLUE, RED, ORANGE]  # Colors for each curve

        NA_real = np.fromfile(r'phd-project\tolerances comparison - NAs.npy')
        tolerances_mirror_lens_mirror = np.fromfile(
            r'phd-project\tolerances comparison - mirror lens mirror.npy').reshape((len(NA_real), 3, 5))
        tolerances_fabry_perot = np.fromfile(r'phd-project\tolerances comparison - fabry perot.npy').reshape(
            (len(NA_real), 2, 4))
        NA_real = NA_real[::10]
        tolerances_mirror_lens_mirror = tolerances_mirror_lens_mirror[::10, 0, 2]
        tolerances_fabry_perot = tolerances_fabry_perot[::10, 0, 2]

        y1 = 1e-7 * (NA / 0.05)**-2
        y2 = 2e-4 * (NA / 0.05)**-0.3

        # Labels
        y_label = Tex(r"Tolerance [rad]", color=TEXT_COLOR, font_size=font_size).rotate(PI / 2).to_edge(LEFT)

        # Axes: log-log scale via manual log10 mapping
        axes = Axes(
            x_range=[np.log10(0.02), np.log10(0.2), 0.2],  # log10 scale for NA (approx 0.03 to 0.2)
            y_range=[-10, -3, 1],  # log10 scale for tolerance
            x_length=6,
            y_length=5,
            axis_config={"include_tip": False, "include_numbers": True},
            y_axis_config={"scaling": LogBase(custom_labels=True)},
            color=TEXT_COLOR,
            x_axis_config={"scaling": LogBase(custom_labels=False), "decimal_number_config":{"num_decimal_places": 2}},
        ).next_to(y_label)

        x_label = Tex(r"Numerical Aperture", color=TEXT_COLOR, font_size=font_size).next_to(axes, DOWN)

        self.add(axes, x_label, y_label)

        # Plot each line
        curve1 = axes.plot_line_graph(NA_real, tolerances_mirror_lens_mirror,
                                      line_color=colors[0], stroke_width=2,
                                      vertex_dot_style=dict(stroke_width=3,  fill_color=PURPLE))

        curve2 = axes.plot_line_graph(NA_real, tolerances_fabry_perot, line_color=colors[1], stroke_width=2,
                                      vertex_dot_style=dict(stroke_width=3,  fill_color=PURPLE))

        self.add(curve1, curve2)  # , curve3, curve4

        # Add title
        title = Tex("Tilt Tolerance", font_size=font_size, color=TEXT_COLOR).next_to(axes, UP)
        self.add(title)

        # Custom legend
        legend_items = [
            ("Fabry-Perot Mirror", curve1, colors[0]),
            ("2-Arms Cavity Mirror", curve2, colors[1]),
        ]

        legend = VGroup()
        for label_text, curve, color in legend_items:
            sample_line = Line(LEFT * 0.4, RIGHT * 0.4, color=color)
            text = Tex(label_text, font_size=font_size, color=TEXT_COLOR)
            item = VGroup(sample_line, text).arrange(RIGHT, buff=0.3)
            legend.add(item)

        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.next_to(axes, RIGHT, buff=0.1)
        self.add(legend)


# manim -pql slides/scene.py Microscope
# manim-slides convert Microscope slides/presentation.html
