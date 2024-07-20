from manim import *
from manim_slides import Slide
import numpy as np
from scipy import special
from typing import Union, Optional, Callable

from PIL import Image
from numpy import asarray
# load the image
# image = Image.open('amos-seminar/Original Paper.png')
# convert image to numpy array
# data = asarray(image)
# save data:
data = np.load('amos-seminar/Original Paper.npy')
# np.save('amos-seminar/Original Paper.npy', data)
# %%
# manim -pql slides/scene.py Microscope --disable_caching
# manim-slides convert Microscope slides/presentation.html

green_color = GREEN
blue_color = BLUE
purple_color = PURPLE
red_color = RED

TRACKER_TIME = ValueTracker(0)
TRACKER_SCANNING_SAMPLE = ValueTracker(0)
TRACKER_SCANNING_CAMERA = ValueTracker(0)
TRACKER_TIME_LASER = ValueTracker(0)
TRACKER_PHASE_MODULATION = ValueTracker(0)
TRACKER_PHASE_MODULATION_SECONDARY = ValueTracker(0)
MICROSCOPE_Y = -0.75
BEGINNING = - 7
FIRST_LENS_X = -2.5
POSITION_LENS_1 = np.array([FIRST_LENS_X, MICROSCOPE_Y, 0])
SECOND_LENS_X = 4
POSITION_LENS_2 = np.array([SECOND_LENS_X, MICROSCOPE_Y, 0])
INITIAL_VERTICAL_LENGTH = 1
FINAL_VERTICAL_LENGTH = 2
POSITION_CAMERA = np.array([SECOND_LENS_X + 2, MICROSCOPE_Y, 0])
END = SECOND_LENS_X + 2
POSITION_WAIST = np.array([(2 * FIRST_LENS_X + SECOND_LENS_X) / 3, MICROSCOPE_Y, 0])
W_0 = 0.14
X_R = (FIRST_LENS_X - SECOND_LENS_X) / 9
W_0_LASER = 0.14
X_R_LASER = 0.5
POSITION_SAMPLE = np.array([-5, MICROSCOPE_Y, 0])
HEIGHT_SAMPLE = INITIAL_VERTICAL_LENGTH
WIDTH_SAMPLE = HEIGHT_SAMPLE * (2 / 3)
HEIGHT_CAMERA = FINAL_VERTICAL_LENGTH
WIDTH_CAMERA = WIDTH_SAMPLE
POSITION_AXES_1 = np.array([-2.2, 2.5, 0])
POSITION_AXES_2 = np.array([2.5, 2.5, 0])
HEIGHT_SCANNING_AXES = 2.5
WIDTH_SCANNING_AXES = 2.5
WAVELENGTH = 0.5
WAVELENGTH_LASER = 0.3
LENGTH_LASER_BEAM = WAVELENGTH_LASER * 6
AXES_RANGE = 1
AMPLITUDE_SIZE = 0.8
PHASE_SHIFT_AMPLITUDE = 0.2
COLOR_INTENSITIES = GREEN
COLOR_UNPERTURBED_AMPLITUDE = GOLD_B
COLOR_PERTURBED_AMPLITUDE = BLUE
COLOR_PHASE_SHIFT_AMPLITUDE = PURPLE_B
COLOR_SCANNING_DOT = GREEN
COLOR_OPTICAL_ELEMENTS = TEAL_E
ZOOM_RATIO = 0.3
POSITION_TITLE = np.array([-6, 2.5, 0])
POSITION_ENERGY_FILTER = (POSITION_CAMERA - WIDTH_CAMERA/2*RIGHT + POSITION_LENS_2 + 0.25 * RIGHT) / 2
VELOCITIES_RATIO = WAVELENGTH_LASER / WAVELENGTH
TITLE_COUNTER = 0
PHASE_OBJECT_SPATIAL_FREQUENCY = 4


def matrix_rgb(mat: np.ndarray):
    return (rgb_to_color(mat[i, :]) for i in range(mat.shape[0]))


def noise_function_1(x):
    return 0.1 * np.sin(3 * x) + 0.2 * np.sin(2 * x)


def noise_function_2(x):
    return 0.1 * np.sin(2 * x) - 0.2 * np.sin(3 * x)

def gaussian_beam_R_x(x, x_R):
    if x == 0:
        return 1000000
    else:
        return x * (1 + x_R ** 2 / x ** 2)


def gaussian_beam_w_x(x, w_0, x_R):
    return w_0 * np.sqrt(1 + (x / x_R) ** 2)


def generate_waves(start_point: Union[np.ndarray, list],
                   end_point: Union[np.ndarray, list],
                   wavelength, width: float,
                   tracker: ValueTracker):
    if isinstance(start_point, list):
        start_point = np.array(start_point, dtype='float64')
    if isinstance(end_point, list):
        end_point = np.array(end_point, dtype='float64')
    path_direction = end_point - start_point
    path_length = np.linalg.norm(path_direction)
    path_direction = path_direction / path_length
    orthogonal_direction = np.cross(path_direction, [0, 0, 1])
    orthogonal_direction[2] = 0  # make sure it's in the xy plane
    orthogonal_direction = orthogonal_direction / np.linalg.norm(orthogonal_direction)
    n_waves = int(path_length // wavelength)
    waves = [Line(
        start=start_point + np.mod(i * wavelength, path_length) * path_direction + width / 2 * orthogonal_direction,
        end=start_point + np.mod(i * wavelength, path_length) * path_direction - width / 2 * orthogonal_direction)
        for i in range(n_waves)]
    waves = VGroup(*waves)
    for i, wave in enumerate(waves):
        wave.add_updater(
            lambda m, i=i: m.move_to(start_point +
                                     np.mod(i * wavelength + tracker.get_value(), path_length) * path_direction)
            .set_opacity(there_and_back_with_pause(
                np.mod(tracker.get_value() + i * wavelength, path_length) / path_length))
        )
    return waves


def dummy_attenuating_function(x: float):
    return np.max([0, 1 - np.abs(x) / 2])


def points_generatoer_gaussian_beam(x: float,
                                    w_0: float,
                                    x_R: float,
                                    center: np.ndarray,
                                    k_vec: np.ndarray,
                                    noise_generator: Callable = lambda t: 0):
    w_x = w_0 * np.sqrt(1 + (x / x_R) ** 2)
    R_x_inverse = x / (x ** 2 + x_R ** 2)
    R_x_inverse = np.tanh(R_x_inverse / 2)  # Renormalizing for the esthetic of the visualization
    transverse_direction = np.cross(k_vec, [0, 0, 1])
    array = np.array([center + x * k_vec - w_x * transverse_direction,
                      center + x * k_vec - w_x * transverse_direction / 2 + R_x_inverse * w_x * k_vec,
                      center + x * k_vec + w_x * transverse_direction / 2 + R_x_inverse * w_x * k_vec,
                      center + x * k_vec + w_x * transverse_direction]) + noise_generator(x) * np.sqrt(
        np.abs((w_x - w_0)) / w_0)

    return array


def points_generator_plane_wave(x: float,
                                center: np.ndarray,
                                k_vec: np.ndarray,
                                width: float,
                                noise_generator: Callable = lambda t: 0):
    transverse_direction = np.cross(k_vec, [0, 0, 1])
    array = np.array([center + x * k_vec - transverse_direction * width / 2,
                      center + x * k_vec - (1 / 3) * transverse_direction * width / 2,
                      center + x * k_vec + (1 / 3) * transverse_direction * width / 2,
                      center + x * k_vec + transverse_direction * width / 2]) + noise_generator(x) * width
    return array


def generate_bazier_wavefront(points: np.ndarray,
                              colors: Optional[Union[np.ndarray, str]] = None,
                              opacities: Optional[np.ndarray] = None, **kwargs):
    if isinstance(colors, str):
        colors = color_to_rgb(colors)
    if colors is not None and opacities is not None:
        colors = (colors.T * opacities).T
    elif colors is None and opacities is not None:
        colors = np.ones((opacities.size, 3))
        colors = (colors.T * opacities).T
    elif colors is None and opacities is None:
        colors = np.ones((1, 3))
    elif colors is not None and opacities is None:
        if colors.ndim == 1:
            colors = colors[np.newaxis, :]
    else:
        raise ValueError(f'Invalid input for:\n{colors=}\nand:\n{opacities=}')

    if points.shape[0] == 2:
        diff = points[1, :] - points[0, :]
        points = np.array([points[0],
                           points[0] + diff / 3,
                           points[0] + diff / 3 * 2,
                           points[1]])

    curve = CubicBezier(points[0, :],
                        points[1, :],
                        points[2, :],
                        points[3, :], **kwargs)
    curve.set_stroke(matrix_rgb(colors)).set_fill(None, opacity=0.0)
    return curve


def generate_bazier_wavefronts(points_generator: Callable,
                               tracker: ValueTracker,
                               wavelength: float,
                               start_parameter: float,
                               end_parameter: float,
                               colors_generator: Callable = lambda t: None,
                               opacities_generator: Callable = lambda t: None,
                               pause_ratio: float = 1.0 / 3, **kwargs):
    length = end_parameter - start_parameter
    n = int(length // wavelength)
    generators = [
        lambda i=i: generate_bazier_wavefront(
            points_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter),
            colors_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter),
            opacities_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter), **kwargs)
        for i in range(n)]
    waves = [always_redraw(generators[i]) for i in range(n)]
    for i, wave in enumerate(waves):
        wave.add_updater(
            lambda m, i=i: m.set_stroke(opacity=there_and_back_with_pause((np.mod(tracker.get_value(), 1) + i) / n,
                                                                          pause_ratio)))
    waves = VGroup(*waves)
    return waves


def generate_wavefronts_start_to_end_gaussian(start_point: Union[np.ndarray, list],
                                              end_point: Union[np.ndarray, list],
                                              tracker: ValueTracker,
                                              wavelength: float,
                                              x_R,
                                              w_0,
                                              center=None,
                                              colors_generator: Callable = lambda t: None,
                                              opacities_generator: Callable = lambda t: None,
                                              noise_generator: Callable = lambda t: 0, **kwargs):
    if isinstance(start_point, list):
        start_point = np.array(start_point)
    if isinstance(end_point, list):
        end_point = np.array(end_point)
    if isinstance(center, list):
        center = np.array(center)
    path = end_point - start_point
    length = np.linalg.norm(path)
    k_vec = path / length
    if center is None:
        center = start_point + length / 2 * k_vec

    start_parameter = k_vec @ (start_point - center)
    end_parameter = k_vec @ (end_point - center)
    points_generator = lambda t: points_generatoer_gaussian_beam(x=t, w_0=w_0, x_R=x_R, center=center, k_vec=k_vec,
                                                                 noise_generator=noise_generator)
    waves = generate_bazier_wavefronts(points_generator=points_generator, tracker=tracker,
                                       wavelength=wavelength, start_parameter=start_parameter,
                                       end_parameter=end_parameter,
                                       colors_generator=colors_generator, opacities_generator=opacities_generator,
                                       pause_ratio=1 / 4, **kwargs)
    return waves


def generate_wavefronts_start_to_end_flat(start_point: Union[np.ndarray, list],
                                          end_point: Union[np.ndarray, list],
                                          tracker: ValueTracker,
                                          wavelength: float,
                                          colors_generator: Callable = lambda t: None,
                                          opacities_generator: Callable = lambda t: None,
                                          noise_generator: Callable = lambda t: 0,
                                          width=1, **kwargs):
    if isinstance(start_point, list):
        start_point = np.array(start_point)
    if isinstance(end_point, list):
        end_point = np.array(end_point)
    path = end_point - start_point
    length = np.linalg.norm(path)
    if width is None:
        width = length / 4
    k_vec = path / length
    center = start_point
    start_parameter = 0
    end_parameter = length
    points_generator = lambda x: points_generator_plane_wave(x=x, center=center, k_vec=k_vec, width=width,
                                                             noise_generator=noise_generator)
    waves = generate_bazier_wavefronts(points_generator, tracker, wavelength=wavelength,
                                       start_parameter=start_parameter, end_parameter=end_parameter,
                                       colors_generator=colors_generator, opacities_generator=opacities_generator,
                                       **kwargs)
    return waves


def generate_scanning_axes(dot_start_point: Union[np.ndarray, list],
                           dot_end_point: Union[np.ndarray, list],
                           axes_position: Union[np.ndarray, list],
                           tracker: ValueTracker,
                           function_to_plot: Callable,
                           axis_x_label: str,
                           axis_y_label: str):
    ax = Axes(x_range=[0, AXES_RANGE, AXES_RANGE / 4], y_range=[0, AXES_RANGE, AXES_RANGE / 4],
              x_length=WIDTH_SCANNING_AXES, y_length=HEIGHT_SCANNING_AXES, tips=False).move_to(axes_position)

    labels = ax.get_axis_labels(
        Tex(axis_x_label).scale(0.5), Tex(axis_y_label).scale(0.5)
    )

    def scanning_dot_generator():
        scanning_dot = Dot(point=dot_start_point + tracker.get_value() * (dot_end_point - dot_start_point),
                           color=COLOR_SCANNING_DOT)
        return scanning_dot

    scanning_dot = always_redraw(scanning_dot_generator)

    def scanning_dot_x_axis_generator():
        scanning_dot_x_axis_start_point = ax.c2p(0, 0)
        scanning_dot_x_axis_end_point = ax.c2p(AXES_RANGE, 0)
        scanning_dot_x_axis = Dot(point=scanning_dot_x_axis_start_point +
                                        tracker.get_value() * (scanning_dot_x_axis_end_point -
                                                               scanning_dot_x_axis_start_point),
                                  color=COLOR_SCANNING_DOT)
        return scanning_dot_x_axis

    scanning_dot_x_axis = always_redraw(scanning_dot_x_axis_generator)

    if function_to_plot is not None:
        amplitude_graph = ax.plot(function_to_plot, color=COLOR_INTENSITIES)
        return ax, labels, scanning_dot, scanning_dot_x_axis, amplitude_graph
    else:
        return ax, labels, scanning_dot, scanning_dot_x_axis

def create_focus_arrow_object(point: np.ndarray):
    return Arrow(start=point + [0.9, 0.9, 0], end=point, color=RED)



# Titles:
title_0 = Tex("k", color=BLACK).to_corner(UL).scale(0.5)
title_1 = Tex("1) Intro").scale(0.5).next_to(title_0, DOWN).align_to(title_0, LEFT)
title_2 = Tex("2) Phase Estimation").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
title_3 = Tex("3) Shot Noise").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
title_4 = Tex("4) Qunatum Metrology").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
title_5 = Tex("5) NOON states").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
title_6 = Tex("6) Multiple Phases Estimation").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
title_7 = Tex("5) Multiple Phases Probe States").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
titles_square = Rectangle(height=title_1.height + 0.1,
                          width=title_1.width + 0.1,
                          stroke_width=2).move_to(title_1.get_center()).set_fill(opacity=0)
y_0 = title_0.get_center()[1]
y_1 = title_1.get_center()[1]
dy = 0.47


# class Microscope(MovingCameraScene, VoiceoverScene):
class Seminar(Slide):  # , ZoomedScene  # MovingCameraScene
    def intro_scene(self):

        title_intro = Tex("Improving phase measurement sensitivity beyond the NOON state", color=WHITE).to_edge(UP).scale(0.7)

        self.wait(1)
        self.next_slide()
        self.play(FadeIn(title_intro))

        self.next_slide()
        abstract_image = ImageMobject(data).scale(0.6)  #
        self.play(FadeIn(abstract_image, shift=DOWN))

        self.next_slide()
        self.play(FadeOut(title_intro, abstract_image))

    def phase_estimation_scene(self):
        title_phase_estimation = Tex("Phase Estimation:", color=WHITE).to_edge(UP)

        introducing_text = Text("When do we need to estimate a phase?", color=WHITE).scale(0.6).next_to(title_phase_estimation, DOWN).shift(1.2*DOWN).to_edge(LEFT)

        first_usage_text = Text("When trying to detect gravitational waves").scale(0.6).next_to(introducing_text, DOWN).to_edge(LEFT).shift(DOWN + RIGHT)
        second_usage_text = Text("When trying to image phase objects").scale(0.6).next_to(first_usage_text, DOWN).align_to(first_usage_text, LEFT).shift(0.4*DOWN)
        third_usage_text = Text("When trying to impress a pretty girl").scale(0.6).next_to(second_usage_text, DOWN).align_to(second_usage_text, LEFT).shift(0.4*DOWN)

        bullet_1 = Dot(color=WHITE).scale(0.3).next_to(first_usage_text, LEFT)
        bullet_2 = Dot(color=WHITE).scale(0.3).next_to(second_usage_text, LEFT)
        bullet_3 = Dot(color=WHITE).scale(0.3).next_to(third_usage_text, LEFT)

        self.play(FadeIn(title_phase_estimation))

        self.next_slide()
        self.play(FadeIn(introducing_text))

        self.next_slide()
        self.play(FadeIn(first_usage_text, bullet_1))

        self.next_slide()
        self.play(FadeIn(second_usage_text, bullet_2))

        self.next_slide()
        self.play(FadeIn(third_usage_text, bullet_3))

        self.next_slide()
        self.play(FadeOut(title_phase_estimation, introducing_text, first_usage_text, second_usage_text, third_usage_text, bullet_1, bullet_2, bullet_3))

    def ligo_scene(self):
        tracker = ValueTracker(0)

        grid = NumberPlane(
            x_length=24*9/16,
            y_length=24,
            x_range=[-12, 12, 1],
            y_range=[-12, 12, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )

        grid.add_updater(lambda x: x.apply_function(lambda p: p + np.array([0.0047 * p[0] * np.cos(tracker.get_value()),
                                                                            -0.0047 * p[1] * np.cos(
                                                                                tracker.get_value()),
                                                                            0])))

        laser_device = Rectangle(color=RED, height=0.3, width=1).shift(3 * LEFT)


        beam_splitter = Rectangle(color=GREEN, height=0.3, width=1).rotate(PI / 4)
        mirror_right = Rectangle(color=BLUE, height=0.3, width=1).rotate(PI / 2).shift(3 * RIGHT)

        mirror_up = Rectangle(color=BLUE, height=0.3, width=1).shift(3 * UP)

        detector = Arc(radius=0.5, angle=PI, color=YELLOW).rotate(PI).shift(2 * DOWN)



        Line_1 = Line(laser_device.get_right(), beam_splitter.get_center(), color=RED)
        Line_1.add_updater(lambda x: x.put_start_and_end_on(laser_device.get_right(), beam_splitter.get_center()))
        Line_2 = Line(beam_splitter.get_center(), mirror_right.get_center(), color=RED)
        Line_2.add_updater(lambda x: x.put_start_and_end_on(beam_splitter.get_center(), mirror_right.get_center()))
        Line_3 = Line(beam_splitter.get_center(), mirror_up.get_center(), color=RED)
        Line_3.add_updater(lambda x: x.put_start_and_end_on(beam_splitter.get_center(), mirror_up.get_center()))
        Line_4 = Line(beam_splitter.get_center(), detector.get_bottom(), color=RED)
        Line_4.add_updater(lambda x: x.put_start_and_end_on(beam_splitter.get_center(), detector.get_bottom()))

        # self.add(grid, laser_device, beam_splitter, mirror_right, mirror_up, detector, Line_1, Line_2, Line_3, Line_4)
        self.play(
            FadeIn(laser_device, beam_splitter, mirror_right, mirror_up, detector, Line_1, Line_2, Line_3, Line_4))
        self.next_slide()
        self.play(Write(grid))
        detector.add_updater(lambda x: x.move_to(grid.c2p(0, -2)))
        mirror_up.add_updater(lambda x: x.move_to(grid.c2p(0, 3)))
        laser_device.add_updater(lambda x: x.move_to(grid.c2p(-3, 0)))
        mirror_right.add_updater(lambda x: x.move_to(grid.c2p(3, 0)))

        tracker.add_updater(lambda mobject, dt: mobject.increment_value(dt))

        self.next_slide()
        self.play(tracker.animate.set_value(2 * PI), run_time=5, rate_func=linear)

        laser_device.clear_updaters()
        mirror_right.clear_updaters()
        mirror_up.clear_updaters()
        detector.clear_updaters()
        grid.clear_updaters()
        Line_1.clear_updaters()
        Line_2.clear_updaters()
        Line_3.clear_updaters()
        Line_4.clear_updaters()

        self.next_slide()
        self.play(
            FadeOut(laser_device, beam_splitter, mirror_right, mirror_up, detector, Line_1, Line_2, Line_3, Line_4,
                    grid))

    def limitations_scene(self):
        title_limitations = Tex("Limitations of phase estimation", color=WHITE).to_edge(UP)
        self.play(FadeIn(title_limitations, shift=DOWN))
        self.next_slide()
        green_color = GREEN
        blue_color = BLUE
        red_color = RED
        alpha = 0.5
        distances_factor = 2

        beam_splitter_r = Rectangle(color=green_color, height=0.3, width=1).shift(distances_factor * RIGHT)
        beam_splitter_l = Rectangle(color=green_color, height=0.3, width=1).shift(distances_factor * LEFT)

        # Shapes
        mirror_u = Rectangle(color=blue_color, height=0.3, width=1).shift(distances_factor * np.tan(alpha) * UP)
        mirror_d = Rectangle(color=blue_color, height=0.3, width=1).shift(distances_factor * np.tan(alpha) * DOWN)

        detector_1 = Arc(radius=0.5, angle=PI, color=YELLOW).rotate(3 * PI / 2 + alpha).move_to(beam_splitter_r).shift(
            distances_factor * (np.cos(alpha) * RIGHT + np.sin(alpha) * UP))
        detector_2 = Arc(radius=0.5, angle=PI, color=YELLOW).move_to(beam_splitter_r).rotate(3 * PI / 2 - alpha).shift(
            distances_factor * (np.cos(alpha) * RIGHT + np.sin(alpha) * DOWN))

        # Beams
        beams = [
            Line(beam_splitter_l.get_center(), mirror_d.get_center(), color=red_color),
            Line(beam_splitter_l.get_center(), mirror_u.get_center(), color=red_color),
            Line(mirror_u.get_center(), beam_splitter_r.get_center(), color=red_color),
            Line(mirror_d.get_center(), beam_splitter_r.get_center(), color=red_color),
            Line(beam_splitter_l.get_center(),
                 beam_splitter_l.get_center() + distances_factor * (np.cos(alpha) * LEFT + np.sin(alpha) * UP),
                 color=red_color),
            Line(beam_splitter_r.get_center(), detector_1.get_center(), color=red_color),
            Line(beam_splitter_r.get_center(), detector_2.get_center(), color=red_color),
        ]

        # Create a filled circle
        circle = Circle(fill_opacity=1, color=BLUE).scale(0.2).move_to(beams[2].get_center()).set_z_index(100)
        phi = MathTex(r"\phi").scale(0.7).move_to(circle.get_center()).set_z_index(100)

        # Labels
        label_1 = Text("1", color=WHITE).scale(0.5).next_to(detector_1, RIGHT)
        label_2 = Text("2", color=WHITE).scale(0.5).next_to(detector_2, RIGHT)

        # Columns
        column_1_v1 = Rectangle(color=BLUE, height=distances_factor * 0.41, width=0.5, fill_color=BLUE,
                             fill_opacity=1).next_to(detector_1, RIGHT, buff=1).align_to(detector_1, DOWN)

        column_2_v1 = Rectangle(color=BLUE, height=distances_factor * 0.59, width=0.5, fill_color=BLUE,
                             fill_opacity=1).next_to(detector_2, RIGHT, buff=1).align_to(detector_2, DOWN)

        column_1_v2 = Rectangle(color=BLUE, height=distances_factor * 0.40, width=0.5, fill_color=BLUE,
                                fill_opacity=1).next_to(detector_1, RIGHT, buff=1).align_to(detector_1, DOWN)

        column_2_v2 = Rectangle(color=BLUE, height=distances_factor * 0.60, width=0.5, fill_color=BLUE,
                                fill_opacity=1).next_to(detector_2, RIGHT, buff=1).align_to(detector_2, DOWN)

        column_1_v3 = Rectangle(color=BLUE, height=distances_factor * 0.46, width=0.5, fill_color=BLUE,
                                fill_opacity=1).next_to(detector_1, RIGHT, buff=1).align_to(detector_1, DOWN)

        column_2_v3 = Rectangle(color=BLUE, height=distances_factor * 0.54, width=0.5, fill_color=BLUE,
                                fill_opacity=1).next_to(detector_2, RIGHT, buff=1).align_to(detector_2, DOWN)

        column_1_v4 = Rectangle(color=BLUE, height=distances_factor * 0.43, width=0.5, fill_color=BLUE,
                                fill_opacity=1).next_to(detector_1, RIGHT, buff=1).align_to(detector_1, DOWN)

        column_2_v4 = Rectangle(color=BLUE, height=distances_factor * 0.57, width=0.5, fill_color=BLUE,
                                fill_opacity=1).next_to(detector_2, RIGHT, buff=1).align_to(detector_2, DOWN)

        label_43_v1 = Text("41", color=WHITE).scale(0.5).next_to(column_1_v1, RIGHT).align_to(detector_1, UP)
        label_57_v1 = Text("59", color=WHITE).scale(0.5).next_to(column_2_v1, RIGHT).align_to(detector_2, UP)

        label_43_v2 = Text("40", color=WHITE).scale(0.5).next_to(column_1_v1, RIGHT).align_to(detector_1, UP)
        label_57_v2 = Text("60", color=WHITE).scale(0.5).next_to(column_2_v1, RIGHT).align_to(detector_2, UP)

        label_43_v3 = Text("46", color=WHITE).scale(0.5).next_to(column_1_v1, RIGHT).align_to(detector_1, UP)
        label_57_v3 = Text("54", color=WHITE).scale(0.5).next_to(column_2_v1, RIGHT).align_to(detector_2, UP)

        label_43_v4 = Text("43", color=WHITE).scale(0.5).next_to(column_1_v1, RIGHT).align_to(detector_1, UP)
        label_57_v4 = Text("57", color=WHITE).scale(0.5).next_to(column_2_v1, RIGHT).align_to(detector_2, UP)

        curved_arrow = CurvedArrow(start_point=label_57_v1.get_right() + DOWN * 2, end_point=label_57_v1.get_right(),
                                   color=WHITE, angle=PI / 2)

        label_shot_noise = Text("Shot noise", color=WHITE).scale(0.5).next_to(curved_arrow, DOWN).shift(0.5 * LEFT)

        classical_behavior = Tex(
            r"Classical shot noise dependency on the number of photons $N$: $\Delta\theta\propto\frac{1}{\sqrt{N}}$").scale(
            0.7).shift(2 * DOWN)

        # Adding to the scene
        self.play(FadeIn(mirror_u, mirror_d, beam_splitter_r, beam_splitter_l, detector_1, detector_2, label_1, label_2), FadeIn(*beams))
        self.play(FadeIn(circle, phi))

        self.next_slide()
        self.play(FadeIn(column_1_v1, column_2_v1, label_43_v1, label_57_v1))

        self.next_slide()
        self.play(Transform(column_1_v1, column_1_v2), Transform(column_2_v1, column_2_v2), Transform(label_43_v1, label_43_v2), Transform(label_57_v1, label_57_v2))

        self.next_slide()
        self.play(Transform(column_1_v1, column_1_v3), Transform(column_2_v1, column_2_v3), Transform(label_43_v1, label_43_v3), Transform(label_57_v1, label_57_v3))

        self.next_slide()
        self.play(Transform(column_1_v1, column_1_v4), Transform(column_2_v1, column_2_v4), Transform(label_43_v1, label_43_v4), Transform(label_57_v1, label_57_v4))

        self.play(FadeIn(curved_arrow, label_shot_noise))

        self.next_slide()
        self.play(FadeIn(classical_behavior))

        self.next_slide()
        self.play(FadeOut(mirror_u, mirror_d, beam_splitter_r, beam_splitter_l, detector_1, detector_2, label_1,
                          label_2, *beams, column_1_v1, column_2_v1, label_43_v1, label_57_v1, curved_arrow,
                          label_shot_noise, classical_behavior, title_limitations, phi, circle))

    def quantum_metrology_scene(self):
        title_quantum_metrology = Tex("Quantum Metrology", color=WHITE).to_edge(UP)
        quantum_metrology_definition = Text("Quantum metrology is the study of utilizing quantum\nproperties of systems (entanglement, squeezing) to improve\nvarious measurement sensitivities",).scale(0.6)

        self.play(FadeIn(title_quantum_metrology))
        self.next_slide()

        self.play(FadeIn(quantum_metrology_definition))
        self.next_slide()
        self.play(FadeOut(title_quantum_metrology), FadeOut(quantum_metrology_definition))

    def noon_state_scene(self):
        title_noon = Text("The NOON state:").to_edge(UP)

        tex_size = 0.8
        intro_text_1 = Tex("The photons number difference and the phase\ndifference are canonical conjugates",
                          color=WHITE).scale(tex_size).next_to(title_noon, DOWN).to_edge(LEFT)
        intro_text_2 = Tex("Therefore - they obey the Heisenberg uncertainty principle:", color=WHITE).scale(tex_size).next_to(intro_text_1, DOWN).align_to(intro_text_1, LEFT)
        intro_text_3 = Tex(r"$\Delta N \Delta \theta \geq \frac{1}{2}$", color=WHITE).scale(tex_size).next_to(intro_text_2, DOWN)
        intro_text_3.move_to([0, intro_text_3.get_y(), 0])
        intro_text_4 = Tex("What is the state with N photons in total that will maximize $\Delta N$?").scale(tex_size).next_to(intro_text_3, DOWN).align_to(intro_text_2, LEFT)

        noon_tex = Tex(r"$\frac{1}{\sqrt{2}}\left(\left|N,0\right\rangle +\left|0,N\right\rangle \right)$").scale(tex_size).next_to(intro_text_4, DOWN)
        noon_tex.move_to([0, noon_tex.get_y(), 0])

        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 1, 1 / 4],
                  x_length=3, y_length=2, tips=False).next_to(noon_tex, DOWN)

        labels = ax.get_axis_labels(
            MathTex(r"N_{2}-N_{1}").scale(0.5), MathTex(r"\mathbb{P}\left(N_{2}-N_{1}\right)").scale(0.5)
        )

        regular_distribution = ax.plot(lambda x: 0.4 * np.exp(-((x / 0.4)**2)), color=TEAL)
        area1_a = ax.get_area(regular_distribution, x_range=(-3, 3), color=TEAL)
        area1_b = ax.get_area(regular_distribution, x_range=(-3, 3), color=TEAL)

        max_var_distribution_a = ax.plot(lambda x: 0.5 * np.exp(-(((x - 2) / 0.1) ** 4)), color=TEAL)
        max_var_distribution_b = ax.plot(lambda x: 0.5 * np.exp(-(((x + 2) / 0.1) ** 4)), color=TEAL)

        area2_a = ax.get_area(max_var_distribution_a, x_range=(-3, 3), color=TEAL)
        area2_b = ax.get_area(max_var_distribution_b, x_range=(-3, 3), color=TEAL)


        self.play(FadeIn(title_noon))

        self.next_slide()
        self.play(FadeIn(intro_text_1))

        self.next_slide()
        self.play(FadeIn(intro_text_2))

        self.next_slide()
        self.play(FadeIn(intro_text_3))

        self.next_slide()
        self.play(FadeIn(intro_text_4))

        self.next_slide()
        self.play(FadeIn(noon_tex))

        self.next_slide()
        self.play(FadeIn(ax, labels))
        self.play(FadeIn(area1_a, area1_b))

        self.next_slide()
        self.play(Transform(area1_a, area2_a), Transform(area1_b, area2_b))

        self.next_slide()
        self.play(FadeOut(title_noon, intro_text_1, intro_text_2, intro_text_3, intro_text_4, noon_tex, ax, labels, area1_a, area1_b))

    def multiple_phases_estimation_sketch_scene(self):
        green_color = GREEN
        blue_color = BLUE
        purple_color = PURPLE
        red_color = RED

        multiple_phases_title = Text(f"Measuring multiple phases", color=WHITE).to_edge(UP)

        # Preparation and Measurement blocks
        preparation_block = Rectangle(color=green_color, height=3, width=2, fill_opacity=1).shift(LEFT * 3)
        measurement_block = Rectangle(color=purple_color, height=3, width=2, fill_opacity=1).shift(RIGHT * 3)

        # Labels for the blocks
        prep_label = Text("Preparation", color=BLACK).scale(0.5).rotate(PI / 2).move_to(preparation_block)
        meas_label = Text("Measurement", color=BLACK).scale(0.5).rotate(PI / 2).move_to(measurement_block)

        vertical_spacing = 1.2
        # Intermediate theta blocks
        theta_blocks = VGroup(
            # Rectangle(color=blue_color, height=0.5, width=0.5, fill_opacity=0, stroke_opacity=0).shift(UP * vertical_spacing),
            Rectangle(color=blue_color, height=0.5, width=0.5, fill_opacity=1),
            Rectangle(color=blue_color, height=0.5, width=0.5, fill_opacity=1).shift(DOWN * vertical_spacing),
        )

        # Labels for theta blocks
        theta_labels = VGroup(
            # Tex(r"\theta_{1}", color=BLACK).scale(0.5).move_to(theta_blocks[0]),
            Tex(r"$\theta_{1}$", color=BLACK).scale(0.5).move_to(theta_blocks[0]),
            Tex(r"$\theta_{d}$", color=BLACK).scale(0.5).move_to(theta_blocks[1]),
        )

        # Dots between the blocks
        dots = VGroup(
            Dot().scale(0.5).move_to((theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2),
            Dot().scale(0.5).move_to(
                (theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2 + vertical_spacing * 0.15 * UP),
            Dot().scale(0.5).move_to(
                (theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2 + vertical_spacing * 0.15 * DOWN)
        )

        # Lines connecting the blocks
        lines = VGroup(
            Line(preparation_block.get_right() + UP * vertical_spacing,
                 measurement_block.get_left() + UP * vertical_spacing, color=red_color),
            Line(preparation_block.get_right(), theta_blocks[0].get_left(), color=red_color),
            Line(preparation_block.get_right() + DOWN * vertical_spacing, theta_blocks[1].get_left(), color=red_color),
            # Line(theta_blocks[0].get_right(), , color=red_color),
            Line(theta_blocks[0].get_right(), measurement_block.get_left(), color=red_color),
            Line(theta_blocks[1].get_right(), measurement_block.get_left() + DOWN * vertical_spacing, color=red_color),
        )

        # Adding everything to the scene
        # self.add(lines)
        # self.add(preparation_block, measurement_block, prep_label, meas_label)
        # self.add(theta_blocks, theta_labels, dots)

        multiple_phase_scheme = VGroup(lines, preparation_block, measurement_block, prep_label, meas_label,
                                       theta_blocks, theta_labels, dots)

        self.play(FadeIn(multiple_phases_title))

        self.next_slide()
        self.play(FadeIn(multiple_phase_scheme))

        self.next_slide(auto_next=True)
        self.play(FadeOut(multiple_phase_scheme, multiple_phases_title))

    def imaging_scene(self):
        focus_arrow = create_focus_arrow_object(point=POSITION_SAMPLE - WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP - 0.15*RIGHT + 0.05*UP)
        incoming_waves = generate_wavefronts_start_to_end_flat(start_point=[BEGINNING, MICROSCOPE_Y, 0],
                                                               end_point=POSITION_SAMPLE,
                                                               tracker=TRACKER_TIME,
                                                               wavelength=WAVELENGTH,
                                                               width=HEIGHT_SAMPLE
                                                               )

        sample_outgoing_waves_opacities = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                                end_point=POSITION_LENS_1,
                                                                                wavelength=WAVELENGTH,
                                                                                width=HEIGHT_SAMPLE,
                                                                                tracker=TRACKER_TIME,
                                                                                opacities_generator=lambda t: np.array(
                                                                                    [1, np.cos(2 * t) ** 2,
                                                                                     np.sin(2 * t) ** 2,
                                                                                     1 - 0.1 * np.cos(t) ** 2]))
        second_lens_outgoing_waves_opacities = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                     end_point=POSITION_CAMERA,
                                                                                     wavelength=WAVELENGTH,
                                                                                     width=HEIGHT_CAMERA,
                                                                                     tracker=TRACKER_TIME,
                                                                                     opacities_generator=lambda
                                                                                         t: np.array(
                                                                                         [1 - 0.1 * np.cos(
                                                                                             (2 - t)) ** 2,
                                                                                          np.sin(2 * (2 - t)) ** 2,
                                                                                          np.cos(2 * (2 - t)) ** 2,
                                                                                          1]))
        gaussian_beam_waves_opacities = generate_wavefronts_start_to_end_gaussian(start_point=POSITION_LENS_1,
                                                                                  end_point=POSITION_LENS_2,
                                                                                  tracker=TRACKER_TIME,
                                                                                  wavelength=WAVELENGTH,
                                                                                  x_R=X_R,
                                                                                  w_0=W_0,
                                                                                  center=POSITION_WAIST,
                                                                                  opacities_generator=lambda
                                                                                      t: np.array([0,
                                                                                                   0.5 + 0.5 * np.cos(
                                                                                                       6 * t + 1) ** 2,
                                                                                                   np.cos(8 * t) ** 2,
                                                                                                   0.2 + 0.8 * np.cos(
                                                                                                       5 * t - 1) ** 2,
                                                                                                   0]))
        lens_1 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=COLOR_OPTICAL_ELEMENTS).move_to(
            POSITION_LENS_1)
        lens_2 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=COLOR_OPTICAL_ELEMENTS).move_to(
            POSITION_LENS_2)
        sample = Rectangle(height=HEIGHT_SAMPLE, width=WIDTH_SAMPLE, color=COLOR_OPTICAL_ELEMENTS).move_to(
            POSITION_SAMPLE)
        camera = Rectangle(height=HEIGHT_CAMERA, width=WIDTH_CAMERA, color=GRAY, fill_color=GRAY_A,
                           fill_opacity=0.3).move_to(POSITION_CAMERA)

        image = ImageMobject(np.uint8([[[250, 100, 80], [100, 40, 32]],
                                       [[20, 8, 6], [100, 40, 32]],
                                       [[0, 0, 0], [220, 108, 36]]])).move_to(POSITION_SAMPLE)
        image.width = WIDTH_SAMPLE
        image.set_z_index(sample.get_z_index() - 1)

        microscope_VGroup = VGroup(incoming_waves, sample, lens_1, sample_outgoing_waves_opacities, lens_2,
                                   gaussian_beam_waves_opacities,
                                   second_lens_outgoing_waves_opacities, camera)
        self.updated_object_animation([microscope_VGroup, image], FadeIn)  # , title_0, title_1, title_2, titles_square
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.next_slide(auto_next=True)

        self.play(FadeIn(focus_arrow, shift=UP / 2, rate_func=smooth),
                  TRACKER_TIME.animate.increment_value(0.5),
                  run_time=1, rate_func=linear)

        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)

        self.next_slide(auto_next=True)
        self.play(focus_arrow.animate.become(create_focus_arrow_object(
            point=POSITION_SAMPLE + HEIGHT_SAMPLE / 2 * UP + 0.05 * UP)), TRACKER_TIME.animate.increment_value(0.5),
                  run_time=1, rate_func=linear)
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.next_slide(auto_next=True)
        self.play(focus_arrow.animate.become(create_focus_arrow_object(
            point=POSITION_SAMPLE + HEIGHT_SAMPLE / 2 * UP + WIDTH_SAMPLE / 2 * RIGHT + 0.05 * UP + 0.05 * RIGHT)),
            TRACKER_TIME.animate.increment_value(0.5), run_time=1, rate_func=linear)
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.next_slide()
        self.play(FadeOut(focus_arrow))

        ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1 = generate_scanning_axes(
            dot_start_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP,
            dot_end_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT - HEIGHT_SAMPLE / 2 * UP,
            axes_position=POSITION_AXES_1,
            tracker=TRACKER_SCANNING_SAMPLE,
            function_to_plot=lambda x: 1 - 0.2 * np.exp(-(6 * (x - 0.7)) ** 2),
            axis_x_label="Position",
            axis_y_label="Intensity")


        self.play(Create(ax_1), Write(labels_1), run_time=2)
        self.play(Create(scanning_dot_1), Create(scanning_dot_x_axis_1))
        # self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(1), Create(amplitude_graph_1), run_time=max(tracker.time_until_bookmark('A'), 2))
        self.play(TRACKER_SCANNING_SAMPLE.animate.increment_value(1), Create(amplitude_graph_1), run_time=2)

        ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2 = generate_scanning_axes(
            dot_start_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP,
            dot_end_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT + HEIGHT_CAMERA / 2 * UP,
            axes_position=POSITION_AXES_2,
            tracker=TRACKER_SCANNING_CAMERA,
            function_to_plot=lambda x: 1 - 0.2 * np.exp(-(6 * (x - 0.7)) ** 2),
            axis_x_label="Position",
            axis_y_label="Intensity")
        camera_scanner_group = VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2)
        self.next_slide()
        self.play(FadeIn(focus_arrow))
        self.play(focus_arrow.animate.become(create_focus_arrow_object(
            point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT + HEIGHT_CAMERA / 2 * UP + 0.05 * UP - 0.15 * RIGHT)))
        self.play(Create(ax_2), Write(labels_2), run_time=2)
        self.play(Create(scanning_dot_2), Create(scanning_dot_x_axis_2))
        self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(amplitude_graph_2), run_time=2)
        self.play(FadeOut(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2,
                                 ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1,
                                 focus_arrow)))

        TRACKER_SCANNING_SAMPLE.set_value(0)

        # ################################################################################################################
        # # Phase object explanation:
        phase_image = ImageMobject(np.uint8([[2, 100], [40, 5], [170, 50]])).move_to(POSITION_SAMPLE)
        phase_image.width = WIDTH_SAMPLE
        phase_image.set_z_index(sample.get_z_index() - 1)
        # self.next_slide()

        sample_outgoing_waves_moises = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                             end_point=POSITION_LENS_1,
                                                                             wavelength=WAVELENGTH,
                                                                             width=HEIGHT_SAMPLE,
                                                                             tracker=TRACKER_TIME,
                                                                             noise_generator=lambda t: np.array(
                                                                                 [[0, 0, 0],
                                                                                  [0.1 * np.sin(2 * np.pi * t), 0, 0],
                                                                                  [0.05 * np.cos(2 * np.pi * t), 0, 0],
                                                                                  [0.1 * np.cos(t), 0, 0]]))
        second_lens_outgoing_waves_moises = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                  end_point=POSITION_CAMERA,
                                                                                  wavelength=WAVELENGTH,
                                                                                  width=HEIGHT_CAMERA,
                                                                                  tracker=TRACKER_TIME,
                                                                                  noise_generator=lambda
                                                                                      t: np.array(
                                                                                      [[0.1 * np.cos(t), 0, 0],
                                                                                       [0.05 * np.cos(2 * np.pi * t), 0,
                                                                                        0],
                                                                                       [0.1 * np.sin(2 * np.pi * t), 0,
                                                                                        0],
                                                                                       [0, 0, 0]]))
        gaussian_beam_waves_moises = generate_wavefronts_start_to_end_gaussian(start_point=POSITION_LENS_1,
                                                                               end_point=POSITION_LENS_2,
                                                                               tracker=TRACKER_TIME,
                                                                               wavelength=WAVELENGTH,
                                                                               x_R=X_R,
                                                                               w_0=W_0,
                                                                               center=POSITION_WAIST,
                                                                               noise_generator=lambda
                                                                                   t: np.array(
                                                                                   [[0.08 * np.sin(5 * t - 1), 0, 0],
                                                                                    [0.09 * np.sin(3 * t + 1), 0, 0],
                                                                                    [0.07 * np.cos(8 * t), 0, 0],
                                                                                    [0, 0, 0]]))


        self.next_slide(auto_next=True)
        self.play(FadeIn(phase_image))
        self.play(sample_outgoing_waves_opacities.animate.become(sample_outgoing_waves_moises),
                  second_lens_outgoing_waves_opacities.animate.become(second_lens_outgoing_waves_moises),
                  gaussian_beam_waves_opacities.animate.become(gaussian_beam_waves_moises), run_time=2,
                  rate_func=linear)
        self.remove(sample_outgoing_waves_opacities, second_lens_outgoing_waves_opacities,
                    gaussian_beam_waves_opacities)
        self.add(sample_outgoing_waves_moises,
                 second_lens_outgoing_waves_moises,
                 gaussian_beam_waves_moises)
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.next_slide()
        microscope_VGroup.add(second_lens_outgoing_waves_moises,
                              sample_outgoing_waves_moises,
                              gaussian_beam_waves_moises)

        ax_complex_amplitude = Axes(x_range=[-AXES_RANGE, AXES_RANGE, 0.25],
                                    y_range=[-AXES_RANGE, AXES_RANGE, 0.25],
                                    x_length=WIDTH_SCANNING_AXES,
                                    y_length=HEIGHT_SCANNING_AXES,
                                    tips=False).move_to(POSITION_AXES_1)
        labels_complex_amplitude = ax_complex_amplitude.get_axis_labels(
            Tex(r'$\text{Re}\left(\psi\right)$').scale(0.3), Tex(r'$\text{Im}\left(\psi\right)$').scale(0.3)
        )

        def circ_complex_amplitude_generator():
            return Circle(
                radius=np.linalg.norm(ax_complex_amplitude.c2p((AMPLITUDE_SIZE, 0)) - ax_complex_amplitude.c2p((0, 0))),
                color=WHITE, stroke_opacity=0.3).move_to(ax_complex_amplitude.c2p(0, 0))

        circ_complex_amplitude = always_redraw(circ_complex_amplitude_generator)

        def arrow_complex_amplitude_generator():
            arrow_complex_amplitude = Line(
                start=ax_complex_amplitude.c2p(0, 0),
                end=ax_complex_amplitude.c2p(AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                    PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                                             AMPLITUDE_SIZE * np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                 PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value()))),
                color=BLUE_B, z_index=ax_complex_amplitude.z_index + 1)
            return arrow_complex_amplitude

        line_complex_amplitude = always_redraw(arrow_complex_amplitude_generator)
        # The dot can not have an always_redraw updater because it is going to change color.
        dot_complex_amplitude = Dot(point=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0), color=COLOR_SCANNING_DOT)
        dot_complex_amplitude.add_updater(lambda m: m.move_to(
            ax_complex_amplitude.c2p(
                AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                    PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                AMPLITUDE_SIZE * np.sin(
                    PHASE_SHIFT_AMPLITUDE * np.sin(
                        PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value())))).set_z_index(
            line_complex_amplitude.z_index + 1))
        self.next_slide()
        self.play(Create(ax_complex_amplitude), Create(labels_complex_amplitude), Create(circ_complex_amplitude))
        # Here it is separated because the dot has to be created after the axes, or it glitches.
        scanning_dot_1.move_to(POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP)
        self.play(Create(line_complex_amplitude), Create(dot_complex_amplitude), Create(scanning_dot_1))
        complex_amplitude_ax_group = VGroup(ax_complex_amplitude, labels_complex_amplitude)
        complex_amplitude_graph_group = VGroup(complex_amplitude_ax_group, circ_complex_amplitude,
                                               line_complex_amplitude, dot_complex_amplitude)
        self.next_slide()
        self.play(TRACKER_SCANNING_SAMPLE.animate.increment_value(1), run_time=4)
        self.next_slide()
        self.updated_object_animation([complex_amplitude_graph_group, microscope_VGroup, scanning_dot_1, phase_image, image], FadeOut)

    def image_while_I_drink(self):
        image_while_I_drink = ImageMobject("amos-seminar/image_while_I_drink.png")
        image_while_I_drink.height = 8

        explanation_text = Text("An action-packed image to keep you\nengaged with the content while I drink", weight=BOLD, color=WHITE, stroke_width=2, stroke_color=BLACK, stroke_opacity=1).to_edge(UP).scale(0.7)

        self.play(FadeIn(image_while_I_drink, explanation_text, shift=20*DOWN), run_time=1)
        self.play(explanation_text.animate.to_edge(DOWN), run_time=10, rate_func=linear)

        self.play(FadeOut(image_while_I_drink, explanation_text, shift=20*UP), run_time=1)

    def multiple_phases_probe_state(self):
        title_noon = Text("The generalized NOON state:").to_edge(UP)
        noon_introducing_text = Text("Let us observe the following state", color=WHITE).next_to(title_noon, DOWN).scale(0.7)
        noon_tex_v1 = MathTex(r"\frac{1}{\sqrt{2}}", r"\left|N,0\right\rangle", "+", r"\frac{1}{\sqrt{2}}", r"\left|0,N\right\rangle").scale(0.7)
        noon_tex_v2 = MathTex(r"\frac{1}{\sqrt{2}}", r"\left|N,0,0,\ldots,0\right\rangle ", "+", r"\frac{1}{\sqrt{2}}", r"\left|0,N,0,\ldots0\right\rangle").scale(0.7)
        noon_tex_v3 = MathTex(r"\frac{1}{\sqrt{2}}", r"\left|N,0,0,\ldots,0\right\rangle ", "+", r"\frac{1}{\sqrt{2}}", r"\left|0,N,0,\ldots0\right\rangle").scale(0.7)
        noon_tex_v4 = MathTex(r"\frac{1}{\sqrt{2}}", r"\left|N,0,0,\ldots,0\right\rangle ", "+", r"\frac{1}{\sqrt{2}}", r"\left(\frac{1}{\sqrt{d}}\left|0,N,0,\ldots0\right\rangle +\frac{1}{\sqrt{d}}\left|0,0,N,\ldots0\right\rangle +\ldots\frac{1}{\sqrt{d}}\left|0,0,0,\ldots N\right\rangle \right)").scale(0.7)
        noon_tex_v5 = MathTex(r"\beta", r"\left|N,0,0,\ldots,0\right\rangle ", "+", r"\alpha", r"\left(\left|0,N,0,\ldots0\right\rangle +\left|0,0,N,\ldots0\right\rangle +\ldots\left|0,0,0,\ldots N\right\rangle \right)").scale(0.7)
        definitions_tex_1 = MathTex(r"d\alpha^2+\beta^2=1").next_to(noon_tex_v5, DOWN).shift(5*LEFT).scale(0.5)
        definitions_tex_2 = Tex(r"$d$ is the number of modes").next_to(definitions_tex_1, RIGHT).shift(0.8*LEFT).scale(0.5)
        definitions_tex_3 = Tex(r"$N$ is the number of particles").next_to(definitions_tex_2, RIGHT).shift(0.8*LEFT).scale(0.5)
        definitions_tex_4 = Tex(r"$\frac{d}{N}$ particles per phase").next_to(definitions_tex_3, RIGHT).shift(0.8*LEFT).scale(0.5)

        derivative_text = Tex(r"Deriving the phase error with respect to $\alpha$ and finding minimum", color=WHITE).next_to(noon_tex_v5, DOWN).shift(0.5*DOWN).scale(0.5)
        alpha_value = MathTex(r"\alpha=\frac{1}{\sqrt{d+\sqrt{d}}}").next_to(derivative_text, DOWN).scale(0.5)
        minimal_error_value_joke_v1 = MathTex(r"\left|\Delta\theta\right|^{2}=", r"?", r"\ ").next_to(alpha_value,DOWN).scale(0.5)
        minimal_error_value_joke_v2 = MathTex(r"\left|\Delta\theta\right|^{2}=", r"5", r"\ ").next_to(alpha_value, DOWN).scale(0.5)
        minimal_error_value_real_v1 = MathTex(r"\left|\Delta\theta\right|^{2}=", r"\frac{\left(1+\sqrt{d}\right)^{2}\frac{d}{4}}{N^{2}}", r"\ ").scale(0.5).next_to(alpha_value, DOWN)
        minimal_error_value_real_v2 = MathTex(r"\left|\Delta\theta\right|^{2}=", r"\frac{\left(1+\sqrt{d}\right)^{2}\frac{d}{4}}{N^{2}}", r"\propto\frac{d^{2}}{N^{2}}").scale(0.5).next_to(alpha_value, DOWN)

        self.play(FadeIn(title_noon))
        self.next_slide()

        self.play(FadeIn(noon_introducing_text))
        self.next_slide()

        self.play(FadeIn(noon_tex_v1))
        self.next_slide()

        self.play(TransformMatchingTex(noon_tex_v1, noon_tex_v2))
        self.next_slide()

        self.play(TransformMatchingTex(noon_tex_v2, noon_tex_v3))
        self.next_slide()

        self.play(TransformMatchingTex(noon_tex_v3, noon_tex_v4))
        self.next_slide()

        self.play(TransformMatchingTex(noon_tex_v4, noon_tex_v5))

        self.next_slide()
        self.play(FadeIn(definitions_tex_1))
        self.play(FadeIn(definitions_tex_2))
        self.play(FadeIn(definitions_tex_3))
        self.play(FadeIn(definitions_tex_4))

        self.next_slide()
        self.play(FadeIn(derivative_text))

        self.next_slide()
        self.play(FadeIn(alpha_value))

        self.next_slide()
        self.play(FadeIn(minimal_error_value_joke_v1))

        self.next_slide()
        self.play(TransformMatchingTex(minimal_error_value_joke_v1, minimal_error_value_joke_v2))

        self.next_slide()
        self.play(TransformMatchingTex(minimal_error_value_joke_v2, minimal_error_value_real_v1))

        self.next_slide()
        self.play(TransformMatchingTex(minimal_error_value_real_v1, minimal_error_value_real_v2))

        self.next_slide()
        self.play(FadeOut(noon_introducing_text,
                          noon_tex_v5,
                          definitions_tex_1,
                          definitions_tex_2,
                          definitions_tex_3,
                          definitions_tex_4,
                          derivative_text,
                          alpha_value),
                  minimal_error_value_real_v2.animate.scale(2).next_to(title_noon, DOWN))

        self.next_slide()
        is_it_good_text = Text("Is it any good?").next_to(minimal_error_value_real_v2, 2*DOWN).scale(0.7)
        self.play(FadeIn(is_it_good_text))

        self.next_slide()
        error_noon = MathTex(r"\left|\Delta\theta_{N00N}\right|^{2}=", r"\frac{d^{3}}{N^{2}}").next_to(is_it_good_text, 2*DOWN)
        self.play(FadeIn(error_noon))

        self.next_slide()
        error_classical = MathTex(r"\left|\Delta\theta_{\text{classical}}\right|^{2}=", r"\frac{d^{2}}{N}").next_to(error_noon, DOWN)
        self.play(FadeIn(error_classical))

        self.next_slide()
        self.play(FadeOut(title_noon, error_noon, error_classical, is_it_good_text, minimal_error_value_real_v2))

    def hb_states_scene(self):
        hb_title = Text("Holland-Burnett States").to_edge(UP).scale(0.8)
        self.play(FadeIn(hb_title))

        problem_text = Text("The Noon state are difficult to realize experimentally.\nIt requires non-linear interactions").next_to(hb_title, DOWN).scale(0.5)

        ideal_solution_text = Text("Solution:").next_to(problem_text, DOWN).scale(0.5).next_to(problem_text, DOWN).align_to(problem_text, LEFT)

        self.play(FadeIn(problem_text))
        self.next_slide()

        self.play(FadeIn(ideal_solution_text))
        self.next_slide()



        # Preparation and Measurement blocks
        preparation_block = Rectangle(color=green_color, height=3, width=2, fill_opacity=1).shift(DOWN + LEFT * 3)
        measurement_block = Rectangle(color=purple_color, height=3, width=2, fill_opacity=1).shift(DOWN + RIGHT * 3)

        # Labels for the blocks
        prep_label = Text("Preparation", color=BLACK).scale(0.5).rotate(PI / 2).move_to(preparation_block)
        meas_label = Text("Measurement", color=BLACK).scale(0.5).rotate(PI / 2).move_to(measurement_block)

        vertical_spacing = 1.2
        # Intermediate theta blocks
        theta_blocks = VGroup(
            # Rectangle(color=blue_color, height=0.5, width=0.5, fill_opacity=0, stroke_opacity=0).shift(UP * vertical_spacing),
            Rectangle(color=blue_color, height=0.5, width=0.5, fill_opacity=1).shift(DOWN),
            Rectangle(color=blue_color, height=0.5, width=0.5, fill_opacity=1).shift(DOWN + DOWN * vertical_spacing),
        )

        # Labels for theta blocks
        theta_labels = VGroup(
            # Tex(r"\theta_{1}", color=BLACK).scale(0.5).move_to(theta_blocks[0]),
            Tex(r"$\theta_{1}$", color=BLACK).scale(0.5).move_to(theta_blocks[0]),
            Tex(r"$\theta_{d}$", color=BLACK).scale(0.5).move_to(theta_blocks[1]),
        )

        # Dots between the blocks
        dots = VGroup(
            Dot().scale(0.5).move_to((theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2),
            Dot().scale(0.5).move_to(
                (theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2 + vertical_spacing * 0.15 * UP),
            Dot().scale(0.5).move_to(
                (theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2 + vertical_spacing * 0.15 * DOWN)
        )

        # Lines connecting the blocks
        lines = VGroup(
            Line(preparation_block.get_right() + UP * vertical_spacing,
                 measurement_block.get_left() + UP * vertical_spacing, color=red_color),
            Line(preparation_block.get_right(), theta_blocks[0].get_left(), color=red_color),
            Line(preparation_block.get_right() + DOWN * vertical_spacing, theta_blocks[1].get_left(), color=red_color),
            # Line(theta_blocks[0].get_right(), , color=red_color),
            Line(theta_blocks[0].get_right(), measurement_block.get_left(), color=red_color),
            Line(theta_blocks[1].get_right(), measurement_block.get_left() + DOWN * vertical_spacing, color=red_color),
        )

        multiple_phase_scheme = VGroup(lines, preparation_block, measurement_block, prep_label, meas_label,
                                       theta_blocks, theta_labels, dots)
        self.play(FadeIn(multiple_phase_scheme))

        new_lines = VGroup(Line(preparation_block.get_left() + UP * vertical_spacing, preparation_block.get_left() + UP * vertical_spacing + LEFT, color=red_color),
                           Line(preparation_block.get_left(), preparation_block.get_left() + LEFT, color=red_color),
                           Line(preparation_block.get_left() + DOWN * vertical_spacing, preparation_block.get_left() + DOWN * vertical_spacing + LEFT, color=red_color)
                           )

        qft_text = Text("QFT", color=BLACK).scale(0.5).rotate(PI / 2).move_to(preparation_block)

        left_creation_operator_up = MathTex(r"\hat{a}_{0}^{\dagger}", r"\ ").next_to(new_lines[0], LEFT).scale(0.5)
        left_creation_operator_mid = MathTex(r"\hat{a}_{1}^{\dagger}", r"\ ").next_to(new_lines[1], LEFT).scale(0.5)
        left_creation_operator_down = MathTex(r"\hat{a}_{d}^{\dagger}", r"\ ").next_to(new_lines[2], LEFT).scale(0.5)
        right_creation_operator_up = MathTex(r"A_{l}^{\dagger}=C\sum_{k}a_{k}^{\dagger}e^{-i2\pi\frac{k \cdot l}{d}}").next_to(lines[0], UP).scale(
            0.5).align_to(lines[2], LEFT)
        self.play(prep_label.animate.become(qft_text), FadeIn(new_lines))

        self.next_slide()
        self.play(FadeIn(left_creation_operator_up, left_creation_operator_mid, left_creation_operator_down))
        self.play(FadeIn(right_creation_operator_up))

        input_state_up_v1 = MathTex(r"\left(\hat{a}_{0}^{\dagger}\right)^{n}", r"\left|0\right\rangle").next_to(new_lines[0], LEFT).scale(0.5)
        input_state_mid_v1 = MathTex(r"\left(\hat{a}_{1}^{\dagger}\right)^{n}", r"\left|0\right\rangle").next_to(new_lines[1], LEFT).scale(0.5)
        input_state_down_v1 = MathTex(r"\left(\hat{a}_{d}^{\dagger}\right)^{n}", r"\left|0\right\rangle").next_to(new_lines[2], LEFT).scale(0.5)

        input_state_up_v2 = MathTex(r"\ ", r"\left|n\right\rangle").next_to(new_lines[0], LEFT).scale(0.5)
        input_state_mid_v2 = MathTex(r"\ ", r"\left|n\right\rangle").next_to(new_lines[1], LEFT).scale(0.5)
        input_state_down_v2 = MathTex(r"\ ", r"\left|n\right\rangle").next_to(new_lines[2], LEFT).scale(0.5)
        # Generate the states in latex:
        for i in range(1, 10):
            input_state_up_v1 = MathTex(r"\left(\hat{a}_{0}^{\dagger}\right)^{n}", r"\left|0\right\rangle").next_to(new_lines[0], LEFT).scale(0.5)
            input_state_mid_v1 = MathTex(r"\left(\hat{a}_{1}^{\dagger}\right)^{n}", r"\left|0\right\rangle").next_to(new_lines[1], LEFT)


        self.next_slide()
        self.play(TransformMatchingTex(left_creation_operator_up, input_state_up_v1),
                  TransformMatchingTex(left_creation_operator_mid, input_state_mid_v1),
                  TransformMatchingTex(left_creation_operator_down, input_state_down_v1),
                  )

        self.next_slide()
        self.play(TransformMatchingTex(input_state_up_v1, input_state_up_v2),
                  TransformMatchingTex(input_state_mid_v1, input_state_mid_v2),
                  TransformMatchingTex(input_state_down_v1, input_state_down_v2),
                  )

        self.next_slide()
        self.play(FadeOut(multiple_phase_scheme, new_lines, left_creation_operator_up, left_creation_operator_mid, left_creation_operator_down,
                          right_creation_operator_up, input_state_up_v2, input_state_mid_v2, input_state_down_v2, problem_text, ideal_solution_text, hb_title))

    def hb_states_results_scene(self):
        results_title = Text("Simulated Results").to_edge(UP).scale(0.8)
        self.play(FadeIn(results_title))

        results_image = ImageMobject("amos-seminar/Results.png").scale(0.7).next_to(results_title, DOWN)

        self.play(FadeIn(results_image))

        self.next_slide()
        self.play(FadeOut(results_title, results_image))

    def conclusion_scene(self):
        question_title = Text("Questions?").to_edge(UP).scale(0.8)
        alexeys_image = ImageMobject("amos-seminar/Alexey-cropped.png")
        alexeys_image.height = 7
        alexeys_image.next_to(question_title, DOWN)
        self.play(FadeIn(question_title))
        self.play(FadeIn(alexeys_image), run_time=60, rate_func=rate_functions.ease_in_circ)

    def construct(self):
        self.intro_scene()

        self.phase_estimation_scene()

        self.ligo_scene()
        #
        self.limitations_scene()

        self.quantum_metrology_scene()

        self.noon_state_scene()

        self.image_while_I_drink()

        self.multiple_phases_estimation_sketch_scene()

        self.imaging_scene()

        self.multiple_phases_probe_state()

        self.hb_states_scene()

        self.hb_states_results_scene()

        self.conclusion_scene()

    def updated_object_animation(self,
                                 objects: Union[Mobject, list[Mobject], VGroup],
                                 animation: Union[Callable, list[Callable]]):
        # This function allows running an animation on objects that are locked by some updater to not perform this
        # animation. For example, if an updater determines an object's opacity, then this object is blocked from being
        # faded in, and this function allows it.
        if isinstance(objects, (list, VGroup)):
            objects = list(objects)
            decomposed_objects = []
            for obj in objects:
                if isinstance(obj, (list, VGroup)):
                    objects.extend(obj)
                else:
                    decomposed_objects.append(obj)
        elif isinstance(objects, Mobject):
            decomposed_objects = [objects]
        else:
            raise TypeError("objects must be Mobject, list[Mobject] or VGroup")

        if isinstance(animation, Callable):
            animation = [animation for i in range(len(decomposed_objects))]

        object_updaters = [obj.get_updaters() for obj in decomposed_objects]
        [obj.clear_updaters() for obj in decomposed_objects]
        self.play(*[a(o) for a, o in zip(animation, decomposed_objects)])
        for i, obj in enumerate(decomposed_objects):
            for updater in object_updaters[i]:
                obj.add_updater(updater)

# if __name__ == "__main__":
#     from manim import *
#     config.media_width = "100%"
#     scene = Seminar()
#     scene.render()

# manim -pql amos-seminar/main.py Seminar
# python -m manim_slides convert Seminar slides/presentation.html



