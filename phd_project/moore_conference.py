from manim import *
import numpy as np
from manim_slides import Slide

from typing import Callable, Union, Optional

def matrix_rgb(mat: np.ndarray):
    return (rgb_to_color(mat[i, :]) for i in range(mat.shape[0]))

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
                              opacities: Optional[np.ndarray] = None, z_index=None, **kwargs):
    if isinstance(colors, (str, utils.color.core.ManimColor)):
        if opacities is not None:
            colors = np.array([color_to_rgb(colors)] * opacities.size)
        else:
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
    if z_index is not None:
        curve.set_z_index(z_index)
    return curve

def generate_bazier_wavefronts(points_generator: Callable,
                               tracker: ValueTracker,
                               wavelength: float,
                               start_parameter: float,
                               end_parameter: float,
                               colors_generator: Callable = lambda t: None,
                               opacities_generator: Callable = lambda t: None,
                               pause_ratio: float = 1.0 / 3,
                               constant_opacity: float = 1,
                               z_index: Optional[int] = None):
    length = end_parameter - start_parameter
    n = int(length // wavelength)
    generators = [
        lambda i=i: generate_bazier_wavefront(
            points_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter),
            colors_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter),
            opacities_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter), z_index=z_index)
        for i in range(n)]
    waves = [always_redraw(generators[i]) for i in range(n)]
    for i, wave in enumerate(waves):
        if constant_opacity is None:
            constant_opacity = 1
        wave.add_updater(
            lambda m, i=i: m.set_stroke(opacity=constant_opacity * there_and_back_with_pause((np.mod(tracker.get_value(), 1) + i) / n,
                                                                          pause_ratio)))
    waves = VGroup(*waves)
    return waves

def generate_wavefronts_start_to_end_flat(start_point: Union[np.ndarray, list],
                                          end_point: Union[np.ndarray, list],
                                          tracker: ValueTracker,
                                          wavelength: float,
                                          colors_generator: Callable = lambda t: None,
                                          opacities_generator: Callable = lambda t: None,
                                          noise_generator: Callable = lambda t: 0,
                                          start_parameter=0.0,
                                          width=1.0, constant_opacity=None, z_index = None, **kwargs):
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
    end_parameter = length
    points_generator = lambda x: points_generator_plane_wave(x=x, center=center, k_vec=k_vec, width=width,
                                                             noise_generator=noise_generator)
    waves = generate_bazier_wavefronts(points_generator, tracker, wavelength=wavelength,
                                       start_parameter=start_parameter, end_parameter=end_parameter,
                                       colors_generator=colors_generator, opacities_generator=opacities_generator, constant_opacity=constant_opacity, z_index=z_index,
                                       **kwargs)
    return waves

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


def generate_coil(center_x, center_y, width, height, spacing, **kwargs):
    """Generate a coil as an ellipse."""
    coil_1 = Ellipse(arc_center=[center_x, center_y + spacing, 0],
                     width=width, height=height,
                     stroke_width=1,
                     **kwargs).set_z_index(40)
    coil_2 = Ellipse(arc_center=[center_x, center_y, 0],
                     width=width, height=height,
                     stroke_width=1,
                     **kwargs).set_z_index(40)
    coil_3 = Ellipse(arc_center=[center_x, center_y - spacing, 0],
                     width=width, height=height,
                     stroke_width=1,
                     **kwargs).set_z_index(40)
    coil = VGroup(coil_1, coil_2, coil_3)
    return coil

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

TRACKER_TIME = ValueTracker(0)
TRACKER_TIME_LASER = ValueTracker(0)
# Top image
SAMPLE_PATH = "phd-project/dummy_image.png"  # change if your image is elsewhere
SAMPLE_WIDTH = 1
SAMPLE_HEIGHT = 0.1
COORDINATE_TOP_IMAGE_X = 0
COORDINATE_TOP_IMAGE_Y = 3
POSITION_SAMPLE = np.array([COORDINATE_TOP_IMAGE_X, COORDINATE_TOP_IMAGE_Y, 0])

# Top coil (above center)
COIL_WIDTH = 3
COIL_SPACING = 0.05
COIL_HEIGHT = 0.2
COORDINATE_COIL_TOP_X = 0
COORDINATE_COIL_TOP_Y = 1.3
POSITION_COIL_TOP = np.array([COORDINATE_COIL_TOP_X, COORDINATE_COIL_TOP_Y, 0])

# Center dot
COORDINATE_CENTER_X = 0
COORDINATE_CENTER_Y = 0
DOT_RADIUS = 0.06

# Bottom coil (below center)
COORDINATE_COIL_BOTTOM_X = 0
COORDINATE_COIL_BOTTOM_Y = -0.8
POSITION_COIL_BOTTOM = np.array([COORDINATE_COIL_BOTTOM_X, COORDINATE_COIL_BOTTOM_Y, 0])

# 90-degree tube (vertical then horizontal to the right)
TUBE_THICKNESS = COIL_WIDTH * 0.6
TUBE_RADIUS_INNER = 0.5
TUBE_RADIUS_OUTER = TUBE_RADIUS_INNER + TUBE_THICKNESS
TUBE_CIRCLE_ORIGIN_X = (TUBE_RADIUS_INNER + TUBE_RADIUS_OUTER) / 2
TUBE_CIRCLE_ORIGIN_Y = COORDINATE_COIL_BOTTOM_Y - 0.5
TUBE_COORDINATE_END = np.array([TUBE_CIRCLE_ORIGIN_X, TUBE_CIRCLE_ORIGIN_Y-(TUBE_RADIUS_INNER + TUBE_RADIUS_OUTER) / 2, 0])
WAVELENGTH = 0.5

# Bottom rectangle
COORDINATE_BOTTOM_RECT_X = 4
COORDINATE_BOTTOM_RECT_Y = TUBE_COORDINATE_END[1]
BOTTOM_RECT_WIDTH = 2.5
BOTTOM_RECT_HEIGHT = 0.4
POSITION_CAMERA = np.array([COORDINATE_BOTTOM_RECT_X, COORDINATE_BOTTOM_RECT_Y, 0])

# Colors
COIL_COLOR = BLUE
TUBE_COLOR = GREY_B
BOTTOM_RECT_COLOR = DARK_GREY
IMAGE_OPACITY = 1.0
COLOR_UNPERTURBED_AMPLITUDE = GOLD_B
COLOR_PERTURBED_AMPLITUDE = BLUE
COLOR_OPTICAL_ELEMENTS = TEAL_E
COLOR_PHASE_SHIFT_AMPLITUDE = PURPLE_B


W_0 = 0.14
X_R = (COORDINATE_COIL_TOP_Y - COORDINATE_COIL_BOTTOM_Y) / 9
W_0_LASER = 0.14
X_R_LASER = 0.5
POSITION_WAIST = np.array([0, (2 * COORDINATE_COIL_TOP_Y + COORDINATE_COIL_BOTTOM_Y) / 3, 0])
WAVELENGTH_LASER = 0.3
LENGTH_LASER_BEAM = WAVELENGTH_LASER * 6
COMMON_TEXT_SCALE = 0.7
COMMONT_TEXT_SPACING = 1

IMAGE_PATH = r'phd_project/dummy_image.png'

class SchematicScene(Slide, MovingCameraScene):
    """A simple schematic: top image, two vertical coils, center dot, 90-degree tube, and bottom rectangle."""

    def construct(self):
        self.wait(1)
        self.next_slide()
        title_0 = Tex("Problem!").scale(COMMON_TEXT_SCALE).shift(2.5*UP)
        title_1 = Tex("Current image acquisition rate of TEM is low due to camera limitations").scale(COMMON_TEXT_SCALE).next_to(title_0, COMMONT_TEXT_SPACING*DOWN)
        title_2 = Tex("Extracting the same amount of information with less electrons reaching the camera").scale(COMMON_TEXT_SCALE).next_to(title_1, COMMONT_TEXT_SPACING*DOWN)
        equivalence_symbol = MathTex(r"\Updownarrow").scale(COMMON_TEXT_SCALE).next_to(title_2, COMMONT_TEXT_SPACING*DOWN)
        title_3 = Tex("Extracting more information per unit time (by increasing current back up)").scale(COMMON_TEXT_SCALE).next_to(equivalence_symbol, COMMONT_TEXT_SPACING*DOWN)
        title_4 = Tex("(Not dose efficiency)").next_to(title_3, COMMONT_TEXT_SPACING*DOWN).scale(COMMON_TEXT_SCALE)
        title_5 = Tex("Are there redundant electrons we can throw away?").next_to(title_4, COMMONT_TEXT_SPACING*DOWN).scale(COMMON_TEXT_SCALE)
        title_6 = Tex("Yes - the local oscillator/DC component of the electron").next_to(title_5, COMMONT_TEXT_SPACING * DOWN).scale(
            COMMON_TEXT_SCALE)

        self.play(FadeIn(title_0), shift=UP)
        self.next_slide()
        self.play(FadeIn(title_1), shift=UP)
        self.next_slide()
        self.play(FadeIn(title_2), shift=UP)
        self.next_slide()
        self.play(FadeIn(equivalence_symbol), shift=UP)
        self.next_slide()
        self.play(FadeIn(title_3), shift=UP)
        self.next_slide()
        self.play(FadeIn(title_4), shift=UP)
        self.next_slide()
        self.play(FadeIn(title_5), shift=UP)
        self.next_slide()
        self.play(FadeIn(title_6), shift=UP)
        self.next_slide()

        self.play(FadeOut(VGroup(title_0, title_1, title_2, equivalence_symbol, title_3, title_4, title_5, title_6)), shift=DOWN)

        # Top image

        # sample = Rectangle(height=SAMPLE_HEIGHT, width=SAMPLE_WIDTH, color=COLOR_OPTICAL_ELEMENTS).move_to(POSITION_SAMPLE)

        phase_image = ImageMobject(np.uint8([[60], [40], [170], [170], [250], [180], [50], [170]]).T).move_to(POSITION_SAMPLE)
        phase_image.width = SAMPLE_WIDTH
        # phase_image.height = SAMPLE_HEIGHT

        # phase_image.set_z_index(sample.get_z_index() - 1)

        # Coils
        coil_top = generate_coil(COORDINATE_COIL_TOP_X, COORDINATE_COIL_TOP_Y, COIL_WIDTH, COIL_HEIGHT, COIL_SPACING, color=COIL_COLOR, fill_opacity=0)
        coil_bottom = generate_coil(COORDINATE_COIL_BOTTOM_X, COORDINATE_COIL_BOTTOM_Y, COIL_WIDTH, COIL_HEIGHT, COIL_SPACING, color=COIL_COLOR, fill_opacity=0)

        # Center dot
        # 90-degree tube (two quarter circles connected by straight lines)
        quarter_circle_top = Arc(radius=TUBE_RADIUS_OUTER, start_angle=PI, angle=PI/2, arc_center=np.array([TUBE_CIRCLE_ORIGIN_X, TUBE_CIRCLE_ORIGIN_Y, 0]), stroke_width=4, color=TUBE_COLOR)
        quarter_circle_bottom = Arc(radius=TUBE_RADIUS_INNER, start_angle=PI, angle=PI/2, arc_center=np.array([TUBE_CIRCLE_ORIGIN_X, TUBE_CIRCLE_ORIGIN_Y, 0]), stroke_width=4, color=TUBE_COLOR)
        tube_group = VGroup(quarter_circle_top, quarter_circle_bottom)

        # Bottom rectangle
        bottom_rect = Rectangle(width=BOTTOM_RECT_HEIGHT, height=BOTTOM_RECT_WIDTH, fill_color=BOTTOM_RECT_COLOR, fill_opacity=1.0, stroke_width=1)
        bottom_rect.move_to(np.array([COORDINATE_BOTTOM_RECT_X, COORDINATE_BOTTOM_RECT_Y, 0]))

        # Add everything to the scene in logical order (background -> foreground)
        self.add(coil_top, coil_bottom, phase_image)
        self.add(bottom_rect)
        # self.add(coil_bottom)
        self.add(tube_group)
        # self.add(coil_top)
        incoming_waves = generate_wavefronts_start_to_end_flat(start_point=[0, 4.5, 0],
                                                               end_point=POSITION_SAMPLE,
                                                               tracker=TRACKER_TIME,
                                                               wavelength=WAVELENGTH,
                                                               width=SAMPLE_WIDTH
                                                               )
        sample_outgoing_unperturbed_waves = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                                  end_point=POSITION_COIL_TOP,
                                                                                  wavelength=WAVELENGTH,
                                                                                  width=SAMPLE_WIDTH,
                                                                                  tracker=TRACKER_TIME,
                                                                                  colors_generator=lambda
                                                                                      t: COLOR_UNPERTURBED_AMPLITUDE)
        sample_outgoing_perturbed_waves_1 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_SAMPLE + 0.2 * RIGHT,
            end_point=POSITION_COIL_TOP - 0.2 * RIGHT,
            wavelength=WAVELENGTH,
            width=SAMPLE_WIDTH,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        sample_outgoing_perturbed_waves_2 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_SAMPLE - 0.2 * RIGHT,
            end_point=POSITION_COIL_TOP + 0.2 * RIGHT,
            wavelength=WAVELENGTH,
            width=SAMPLE_WIDTH,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

        gaussian_beam_waves_unperturbed = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_COIL_TOP,
            end_point=POSITION_COIL_BOTTOM,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST,
            colors_generator=lambda
                t: COLOR_UNPERTURBED_AMPLITUDE)
        gaussian_beam_waves_perturbed_1 = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_COIL_TOP + W_0 * RIGHT,
            end_point=POSITION_COIL_BOTTOM + 4 * W_0 * RIGHT,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST + 2.3 * W_0 * RIGHT,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        gaussian_beam_waves_perturbed_2 = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_COIL_TOP - W_0 * RIGHT,
            end_point=POSITION_COIL_BOTTOM - 4 * W_0 * RIGHT,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST - 2.3 * W_0 * RIGHT,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

        second_lens_outgoing_waves_unperturbed = generate_wavefronts_start_to_end_flat(start_point=POSITION_COIL_BOTTOM,
                                                                                       end_point=POSITION_COIL_BOTTOM + 1*DOWN,
                                                                                       wavelength=WAVELENGTH,
                                                                                       width=BOTTOM_RECT_WIDTH * 0.7,
                                                                                       tracker=TRACKER_TIME,
                                                                                       colors_generator=lambda
                                                                                           t: COLOR_UNPERTURBED_AMPLITUDE)
        second_lens_outgoing_waves_purterbed_1 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_COIL_BOTTOM - 0.1 * RIGHT,
            end_point=POSITION_COIL_BOTTOM + 1*DOWN + 0.1*RIGHT,
            wavelength=WAVELENGTH,
            width=BOTTOM_RECT_WIDTH * 0.7,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        second_lens_outgoing_waves_purterbed_2 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_COIL_BOTTOM + 0.1 * RIGHT,
            end_point=POSITION_COIL_BOTTOM + 1*DOWN - 0.1*RIGHT,
            wavelength=WAVELENGTH,
            width=BOTTOM_RECT_WIDTH * 0.7,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

        energy_filter_waves_unperturbed = generate_wavefronts_start_to_end_flat(start_point=TUBE_COORDINATE_END,
                                                                                end_point=POSITION_CAMERA + 0.1 * LEFT,
                                                                                wavelength=WAVELENGTH,
                                                                                width=BOTTOM_RECT_WIDTH * 0.7,
                                                                                tracker=TRACKER_TIME,
                                                                                constant_opacity=0.4,
                                                                                colors_generator=lambda
                                                                                   t: COLOR_UNPERTURBED_AMPLITUDE)
        energy_filter_waves_perturbed_1 = generate_wavefronts_start_to_end_flat(
            start_point=TUBE_COORDINATE_END - 0.3 * UP,
            end_point=POSITION_CAMERA + 0.1 * LEFT + 0.3*UP,
            wavelength=WAVELENGTH,
            width=BOTTOM_RECT_WIDTH * 0.7,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        energy_filter_waves_perturbed_2 = generate_wavefronts_start_to_end_flat(
            start_point=TUBE_COORDINATE_END + 0.3 * UP,
            end_point=POSITION_CAMERA + 0.1 * LEFT - 0.3 * UP,
            wavelength=WAVELENGTH,
            width=BOTTOM_RECT_WIDTH * 0.7,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

        laser_tilt = 0
        laser_waves = generate_wavefronts_start_to_end_gaussian(
                start_point=POSITION_WAIST + np.cos(laser_tilt) * LENGTH_LASER_BEAM * RIGHT + np.sin(laser_tilt) * LENGTH_LASER_BEAM * UP,
                end_point=POSITION_WAIST - np.cos(laser_tilt) * LENGTH_LASER_BEAM * RIGHT - np.sin(laser_tilt) * LENGTH_LASER_BEAM * UP,
                tracker=TRACKER_TIME_LASER,
                wavelength=WAVELENGTH_LASER,
                x_R=X_R_LASER,
                w_0=W_0_LASER,
                center=POSITION_WAIST,
                colors_generator=lambda t: RED)
        waves_vgroup = VGroup(incoming_waves,
                              sample_outgoing_unperturbed_waves,
                              sample_outgoing_perturbed_waves_1,
                              sample_outgoing_perturbed_waves_2,
                              gaussian_beam_waves_unperturbed,
                              gaussian_beam_waves_perturbed_1,
                              gaussian_beam_waves_perturbed_2,
                              second_lens_outgoing_waves_unperturbed,
                              second_lens_outgoing_waves_purterbed_1,
                              second_lens_outgoing_waves_purterbed_2,
                              energy_filter_waves_unperturbed,
                              energy_filter_waves_perturbed_1,
                              energy_filter_waves_perturbed_2,
                              laser_waves)
        # focus_arrow = create_focus_arrow_object(point=POSITION_WAIST + 0.1 * RIGHT - 0.07 * RIGHT)
        self.updated_object_animation(waves_vgroup, FadeIn)

        self.wait(1)
        # self.add(incoming_waves,
        #                                  sample_outgoing_unperturbed_waves,
        #                                sample_outgoing_perturbed_waves_1,
        #                                sample_outgoing_perturbed_waves_2,
        #                                gaussian_beam_waves_unperturbed,
        #                                gaussian_beam_waves_perturbed_1,
        #                                gaussian_beam_waves_perturbed_2,
        #                                second_lens_outgoing_waves_unperturbed,
        #                                second_lens_outgoing_waves_purterbed_1,
        #                                second_lens_outgoing_waves_purterbed_2,
        #                                laser_waves)
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)

        # Short pause so manim preview shows the static frame
        # self.wait(1)

        # zoom in on the center dot
        self.camera.frame.save_state()
        ZOOM_RATIO = 10

        self.next_slide()
        self.updated_object_animation(waves_vgroup, FadeOut, added_animation=[self.camera.frame.animate.set(width=14.222 / ZOOM_RATIO).move_to(POSITION_WAIST + 0.2 * LEFT)])

        laser_tilt = np.pi / 6
        laser_spacing = 0.2
        laser_velocity = 1 * laser_spacing
        dots_spacing = laser_spacing / np.sin(laser_tilt) / 2
        dots_velocity = laser_velocity / np.sin(laser_tilt) * 2
        laser_global_shift = laser_spacing * 2 * 1/8
        # dots_tracker = ValueTracker(0)
        laser_lines_1 = generate_wavefronts_start_to_end_flat(start_point=POSITION_WAIST + LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
                                          end_point = POSITION_WAIST - LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
                                          tracker=TRACKER_TIME,
                                          wavelength=laser_spacing * 2,
                                          start_parameter = laser_global_shift,
                                          colors_generator= lambda t: RED,
                                          opacities_generator = lambda t: np.array([0, 1, 1, 0]),
                                          width=0.5,
        z_index=0)

        laser_lines_2 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_WAIST + LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
            end_point=POSITION_WAIST - LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
            start_parameter=laser_spacing * 2 * 1/2 + laser_global_shift,
            tracker=TRACKER_TIME,
            wavelength=laser_spacing * 2,
            colors_generator=lambda t: RED,
            width=0.5, opacities_generator=lambda t: np.array([0, 0.2, 0.2, 0]),
        z_index=0)

        laser_lines_3 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_WAIST + LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
            end_point=POSITION_WAIST - LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
            start_parameter=-laser_spacing * 2 * 1/4 + laser_global_shift,
            tracker=TRACKER_TIME,
            wavelength=laser_spacing * 2,
            colors_generator=lambda t: RED,
            width=0.5, opacities_generator=lambda t: np.array([0, 0.5, 0.5, 0]),
        z_index=0)

        laser_lines_4 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_WAIST + LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
            end_point=POSITION_WAIST - LENGTH_LASER_BEAM * np.array([np.cos(laser_tilt), np.sin(laser_tilt), 0]),
            start_parameter=laser_spacing * 2 * 1 / 4 + laser_global_shift,
            tracker=TRACKER_TIME,
            wavelength=laser_spacing * 2,
            colors_generator=lambda t: RED,
            width=0.5, opacities_generator=lambda t: np.array([0, 0.5, 0.5, 0]),
        z_index=0)

        # Plot a train of dots that are moving in a constant speed downwards at the center of the screen:

        dots = VGroup(*[Dot(point=POSITION_WAIST + i * DOWN * dots_spacing, radius=0.02, color=COLOR_PHASE_SHIFT_AMPLITUDE) for i in range(32)])
        # put the dots on the top layer:
        dots.set_z_index(100)
        dots.add_updater(
            lambda m: m.move_to(POSITION_WAIST + (TRACKER_TIME.get_value()) * DOWN * dots_velocity))
        self.updated_object_animation([laser_lines_1, laser_lines_2, laser_lines_3, laser_lines_4], FadeIn, added_animation=[FadeIn(dots)])
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=8, rate_func=linear)
        self.next_slide()

        # --- Add small axes with unit circle, moving dot and connecting line on the left third of the view ---
        # Place axes on the left third of the current camera frame (camera was zoomed above).
        frame_center = self.camera.frame.get_center()
        frame_width = self.camera.frame.get_width()
        # left third offset from frame center
        axes_center = frame_center + (-frame_width / 3) * RIGHT + 0.05 * DOWN

        axes_size = 5
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=axes_size,
            y_length=axes_size,
            axis_config={"include_ticks": True, "stroke_width": 0.1, "include_tip": False},
        )
        axes.move_to(axes_center).scale(1/ZOOM_RATIO)
        bg_square = Square(side_length=axes_size, fill_color=BLACK, fill_opacity=1.0, stroke_width=0).scale(1/ZOOM_RATIO)
        bg_square.move_to(axes_center)
        # ensure background square sits above laser artifacts but behind axes and markers
        bg_square.set_z_index(10)
        axes.set_z_index(20)
        # unit circle in axes coordinates (radius = 1 * axes unit)
        unit_radius = axes.x_axis.unit_size
        unit_circle = Circle(radius=unit_radius / ZOOM_RATIO, color=TEAL, stroke_width=0.5).move_to(axes.c2p(0, 0)).set_z_index(30)

        # Use the same dots_tracker from above. Theta(t) = pi/2 + A * sin(w * t)
        A = np.pi / 6
        w = 2 * np.pi
        phi = 0
        def theta(t):
            return np.pi / 2 + A * np.cos(w * t + phi)

        # initial dot position
        t0 = TRACKER_TIME.get_value()
        dot_pos = axes.c2p(np.cos(theta(t0)), np.sin(theta(t0)))
        moving_dot = Dot(point=dot_pos, radius=0.01, color=COLOR_PHASE_SHIFT_AMPLITUDE).set_z_index(30).set_z_index(30)
        # updater to follow (cos(theta(t)), sin(theta(t))) with t = dots_tracker.get_value()
        moving_dot.add_updater(lambda m: m.move_to(axes.c2p(
            np.cos(theta(TRACKER_TIME.get_value())),
            np.sin(theta(TRACKER_TIME.get_value()))
        )))

        # line from origin (0,0) to the moving dot
        line_to_dot = always_redraw(lambda: Line(axes.c2p(0, 0), moving_dot.get_center(), color=YELLOW, stroke_width=0.1)).set_z_index(30)

        single_frequency_laser_tex = Tex(r"Monochromatic laser: $\psi\rightarrow\psi\cdot e^{i\frac{\pi}{2}}$").scale(
            0.8 / ZOOM_RATIO)
        double_frequency_laser_tex = Tex(
            r"Bichromatic laser: $\psi\rightarrow\psi\cdot e^{i\left(\frac{\pi}{2}+A\sin\left(\omega_{\text{beating}}t\right)\right)}$",
            r"$=e^{i\frac{\pi}{2}}\cdot\sum_{q\in\mathbb{Z}}a_{n}\cdot\psi\cdot e^{i\omega_{n}t}$").scale(
            0.8 / ZOOM_RATIO)
        single_frequency_laser_tex[0][21].set_color(COLOR_UNPERTURBED_AMPLITUDE)
        single_frequency_laser_tex[0][23:].set_color(COLOR_PHASE_SHIFT_AMPLITUDE)
        double_frequency_laser_tex[0][18].set_color(COLOR_UNPERTURBED_AMPLITUDE)
        double_frequency_laser_tex[0][21:].set_color(COLOR_PHASE_SHIFT_AMPLITUDE)
        double_frequency_laser_tex[1][1:].set_color(COLOR_PHASE_SHIFT_AMPLITUDE)
        double_frequency_laser_tex.next_to(axes, 0.1 * UP).align_to(axes, LEFT).shift(0.05 * RIGHT)
        single_frequency_laser_tex.next_to(double_frequency_laser_tex, 0.1 * UP).align_to(double_frequency_laser_tex,
                                                                                          LEFT)
        # bg = SurroundingRectangle(double_frequency_laser_tex[1], color=BLACK, fill_color=BLACK, fill_opacity=0.5)
        # bg.set_z_index(15)

        axes_vgroup = VGroup(axes, unit_circle, line_to_dot, moving_dot, single_frequency_laser_tex, double_frequency_laser_tex)
        # Add to scene and play tracker increment so the dot moves
        self.add(axes, unit_circle, line_to_dot, moving_dot)
        # animate dots_tracker so the dot executes the motion; adjust increment/run_time as desired
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=8, rate_func=linear)

        self.next_slide()
        self.play(FadeIn(single_frequency_laser_tex), run_time=2)
        self.next_slide()
        self.play(FadeIn(double_frequency_laser_tex[0]), run_time=2)
        self.next_slide()
        self.updated_object_animation([laser_lines_1, laser_lines_2, laser_lines_3, laser_lines_4, dots], FadeOut, added_animation=[FadeIn(double_frequency_laser_tex[1])])
        self.next_slide()
        # zoom out:
        # self.play(, run_time=2)
        waves_vgroup.remove(laser_waves)
        self.updated_object_animation(waves_vgroup, FadeIn, added_animation=[Restore(self.camera.frame), FadeOut(axes_vgroup)])
        self.updated_object_animation([laser_lines_1, laser_lines_2, laser_lines_3, laser_lines_4], FadeIn)
        self.next_slide(loop=True)
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=8, rate_func=linear)

    def updated_object_animation(self,
                                 objects: Union[Mobject, list[Mobject], VGroup],
                                 animation: Union[Callable, list[Callable]],
                                 added_animation = None):
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
        if added_animation is None:
            self.play(*[a(o) for a, o in zip(animation, decomposed_objects)])
        else:
            self.play(*[a(o) for a, o in zip(animation, decomposed_objects)], *added_animation)
        for i, obj in enumerate(decomposed_objects):
            for updater in object_updaters[i]:
                obj.add_updater(updater)

# a = SchematicScene()
# a.construct()
# %%
# manim -pql phd_project/moore_conference.py SchematicScene
# manim-slides convert SchematicScene slides/moore.pptx