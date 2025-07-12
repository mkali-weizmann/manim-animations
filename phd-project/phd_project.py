from manim import *
# from manim_slides import Slide
import numpy as np
from scipy import special
from typing import Union, Optional, Callable
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.azure import AzureService

# manim -pql slides/scene.py Microscope --disable_caching
# manim-slides convert Microscope slides/presentation.html


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
    if isinstance(colors, (str, utils.color.core.ManimColor)):
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


class Microscope(MovingCameraScene, VoiceoverScene):
# class Microscope(Slide, MovingCameraScene):  # , ZoomedScene
    def construct(self):
        self.set_speech_service(GTTSService(transcription_model='base'))
        # self.set_speech_service(
        #     AzureService(
        #         voice="en-US-AriaNeural",
        #         style="friendly",#"newscast-casual",
        #         global_speed=1.07
        #     )
        # )
        ################################################################################################################
        # Intro titles:
        #
        self.wait(1)
        # self.next_slide()
        title_0 = Tex("kasdasd", color=BLACK).to_corner(UL).scale(0.5)
        title_1 = Tex("1) Microscope").scale(0.5).next_to(title_0, DOWN).align_to(title_0, LEFT)
        title_2 = Tex("2) Phase Object").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        title_3 = Tex("3) Waves Decomposition").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        title_4 = Tex("4) Phase Mask").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        title_5 = Tex("5) Phase Mask + Attenuation").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        titles_square = Rectangle(height=title_1.height + 0.1,
                                  width=title_1.width + 0.1,
                                  stroke_width=2).move_to(title_1.get_center()).set_fill(opacity=0)
        y_0 = title_0.get_center()[1]
        y_1 = title_1.get_center()[1]
        dy = 0.47
        focus_arrow = create_focus_arrow_object(point=POSITION_SAMPLE - WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP - 0.15*RIGHT + 0.05*UP)
        bad_title = Tex("Transmission Electron Microscope image enhancement\nusing free electron-photon ponderomotive interaction",  color=WHITE).scale(0.75)
        good_title = Tex("Shooting laser on electrons\nmakes images good", color=WHITE).scale(0.75)
        with self.voiceover(
                text="Today we are going to talk about Transmission Electron Microscope image enhancement, using free electron-photon ponderomotive interaction.") as tracker:  #
            self.wait(1)
            self.play(FadeIn(bad_title, shift=DOWN))
        # self.next_slide()
        with self.voiceover(
                text="This name is not very catchy. Simply speaking, we are going to see how <bookmark mark='A'/> Shooting laser on electrons makes images good.<bookmark mark='B'/>") as tracker:  #
            self.wait_until_bookmark("A")
            self.play(FadeOut(bad_title, shift=DOWN), FadeIn(good_title, shift=DOWN))
            # self.next_slide()
            self.wait_until_bookmark("B")
            self.play(FadeOut(good_title, shift=DOWN))

        ###############################################################################################################
        # Microscope introduction:
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
        lens_1 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=COLOR_OPTICAL_ELEMENTS).move_to(POSITION_LENS_1)
        lens_2 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=COLOR_OPTICAL_ELEMENTS).move_to(POSITION_LENS_2)
        sample = Rectangle(height=HEIGHT_SAMPLE, width=WIDTH_SAMPLE, color=COLOR_OPTICAL_ELEMENTS).move_to(POSITION_SAMPLE)
        camera = Rectangle(height=HEIGHT_CAMERA, width=WIDTH_CAMERA, color=GRAY, fill_color=GRAY_A,
                           fill_opacity=0.3).move_to(POSITION_CAMERA)
        energy_filter = Rectangle(height=HEIGHT_CAMERA, width=WIDTH_CAMERA/5, color=GREEN_E, fill_color=GREEN_C,
                           fill_opacity=0.3).move_to(POSITION_ENERGY_FILTER)

        image = ImageMobject(np.uint8([[[250, 100, 80], [100, 40, 32]],
                                             [[20, 8, 6], [100, 40, 32]],
                                             [[0, 0, 0], [220, 108, 36]]])).move_to(POSITION_SAMPLE)
        image.width = WIDTH_SAMPLE
        image.set_z_index(sample.get_z_index() - 1)

        microscope_VGroup = VGroup(incoming_waves, sample, lens_1, sample_outgoing_waves_opacities, lens_2,
                                   gaussian_beam_waves_opacities,
                                   second_lens_outgoing_waves_opacities, camera)
        with self.voiceover(
                text="""Let's start by seeing how does a transmission microscope work.
                This explanation applies both for a regular optical microscopes, and for electron microscopes.
                <bookmark mark='A'/> An incoming plane wave is approaching the sample we want to image from the left.
                <bookmark mark='B'/> The sample scatters or absorbs parts of the wave, and so
                <bookmark mark='C'/> the intensity of the wave after the sample changes accordingly.""") as tracker:
            self.updated_object_animation([microscope_VGroup, title_0, title_1, title_2, titles_square, image], FadeIn)
            self.play(TRACKER_TIME.animate.increment_value(tracker.time_until_bookmark('A')), run_time=tracker.time_until_bookmark('A'), rate_func=linear)
            # self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
            self.play(FadeIn(focus_arrow, shift=UP/2, rate_func=smooth),
                      TRACKER_TIME.animate.increment_value(1),
                      run_time=1, rate_func=linear)
            # self.next_slide(loop=True)
            self.play(TRACKER_TIME.animate.increment_value(tracker.time_until_bookmark('B')),
                      run_time=tracker.time_until_bookmark('B'), rate_func=linear)
            # self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
            # self.next_slide()
            self.play(focus_arrow.animate.become(create_focus_arrow_object(point=POSITION_SAMPLE + HEIGHT_SAMPLE / 2 * UP + 0.05 * UP)),
                      TRACKER_TIME.animate.increment_value(1),
                      run_time=1, rate_func=linear)
            # self.next_slide(loop=Ture)
            self.play(TRACKER_TIME.animate.increment_value(tracker.time_until_bookmark('C')),
                      run_time=tracker.time_until_bookmark('C'), rate_func=linear)
            # self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
            # self.next_slide()
            self.play(focus_arrow.animate.become(
                create_focus_arrow_object(point=POSITION_SAMPLE + HEIGHT_SAMPLE / 2 * UP + WIDTH_SAMPLE / 2 * RIGHT + 0.05 * UP + 0.05*RIGHT)),
                      TRACKER_TIME.animate.increment_value(1),
                      run_time=1, rate_func=linear)
            # self.next_slide(loop=True)
            self.play(TRACKER_TIME.animate.increment_value(tracker.get_remaining_duration()-1),
                      run_time=tracker.get_remaining_duration()-1, rate_func=linear)
            # self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
            # self.next_slide()
            self.play(FadeOut(focus_arrow))

        ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1 = generate_scanning_axes(
            dot_start_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP,
            dot_end_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT - HEIGHT_SAMPLE / 2 * UP,
            axes_position=POSITION_AXES_1,
            tracker=TRACKER_SCANNING_SAMPLE,
            function_to_plot=lambda x: 1 - 0.2 * np.exp(-(6 * (x - 0.7)) ** 2),
            axis_x_label="Position",
            axis_y_label="Intensity")

        with self.voiceover(
                text="""If we measured the intensity of the field right after the sample. The
                Intensity as a function of position would be our image.
                <bookmark mark='A'/>  However, the sample is small, and so we can not probe the field at such
                small scales.""") as tracker:
            self.play(Create(ax_1), Write(labels_1), run_time=2)
            self.play(Create(scanning_dot_1), Create(scanning_dot_x_axis_1))
            # self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(1), Create(amplitude_graph_1), run_time=max(tracker.time_until_bookmark('A'), 2))
            self.play(TRACKER_SCANNING_SAMPLE.animate.increment_value(1), Create(amplitude_graph_1), run_time=2)
            # # END INDENTATION
        # self.next_slide()

        ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2 = generate_scanning_axes(
            dot_start_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP,
            dot_end_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT + HEIGHT_CAMERA / 2 * UP,
            axes_position=POSITION_AXES_2,
            tracker=TRACKER_SCANNING_CAMERA,
            function_to_plot=lambda x: 1 - 0.2 * np.exp(-(6 * (x - 0.7)) ** 2),
            axis_x_label="Position",
            axis_y_label="Intensity")
        camera_scanner_group = VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2)
        with self.voiceover(
                text="""Therefore, we use optics elements to take the field and magnify it as it is.
                        The magnified field hits the camera which measures it's intensity.
                        <bookmark mark='A'/> Since the optics copy the field exactly as it is, the intensity pattern
                        on the camera equals exactly to the pattern right after the sample""") as tracker:
            self.play(FadeIn(focus_arrow))
            # self.next_slide()
            self.play(focus_arrow.animate.become(create_focus_arrow_object(point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT + HEIGHT_CAMERA / 2 * UP + 0.05 * UP - 0.15*RIGHT)))
            # self.next_slide()
            self.play(Create(ax_2), Write(labels_2),  run_time=2)
            # self.next_slide()
            self.play(Create(scanning_dot_2), Create(scanning_dot_x_axis_2))
            self.wait_until_bookmark('A')
            self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(amplitude_graph_2), run_time=tracker.get_remaining_duration())
            # self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(amplitude_graph_2), run_time=2)
            # self.next_slide()
            self.play(FadeOut(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2,
                                     ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1, focus_arrow)))

            TRACKER_SCANNING_SAMPLE.set_value(0)
            # END INDENTATION

        # ################################################################################################################
        # # Phase object explanation:
        phase_image = ImageMobject(np.uint8([[2, 100], [40, 5], [170, 50]])).move_to(POSITION_SAMPLE)
        phase_image.width = WIDTH_SAMPLE
        phase_image.set_z_index(sample.get_z_index() - 1)
        # self.next_slide()
        self.play(FadeOut(title_0, shift=dy * UP),
                  title_1.animate.move_to([title_1.get_center()[0], y_0, 0]),
                  title_2.animate.move_to([title_2.get_center()[0], y_1, 0]),
                  FadeIn(title_3, shift=dy * UP),
                  titles_square.animate.set_width(title_2.width + 0.1).move_to([title_2.get_center()[0], y_1, 0]))
        self.play(FadeOut(image))


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

        with self.voiceover(
                text="""It is not always that easy. Sometimes, objects we want to image do not scatter
                or absorb the incoming wave, but only delay it a bit. you can think of imaging those objects like
                taking a picture of a clear glass on a white background. The light travels slower in the glass, but the glass is transparent
                and hard to see. Such objects are called phase objects and are common in electron microscopy.
                Since the wave slows down in the object it acquires more phase when passing through it and the wavefronts are distorted.""") as tracker:

            # self.next_slide()
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
            # self.next_slide(loop=True)
            # self.play(TRACKER_TIME.animate.increment_value(tracker.get_remaining_duration()),
            #           run_time=tracker.get_remaining_duration(), rate_func=linear)
            self.play(TRACKER_TIME.animate.increment_value(tracker.get_remaining_duration()),
                      run_time=tracker.get_remaining_duration(), rate_func=linear)
            # self.next_slide()
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
                AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                AMPLITUDE_SIZE * np.sin(
                    PHASE_SHIFT_AMPLITUDE * np.sin(PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value())))).set_z_index(
            line_complex_amplitude.z_index + 1))
        self.wait(1)
        with self.voiceover(
                text="""Let's observe the effect of the phase object on the wave in the complex plane of amplitude.
                If we could scan the field right after the object we would see the phase delay induced by the sample <bookmark mark='A'/> as a rotation in the complex plane.
                Since the absolute value of the amplitude does not change when being rotated, the intensity is constant across the wave.
                <bookmark mark='B'/> And what would be the resulting image?
                <bookmark mark='C'/> Since the camera is sensitive only to intensity,
                The resulting image will be a constant white . <bookmark mark='D'/> we lost all contrast!""") as tracker:
            self.play(Create(ax_complex_amplitude), Create(labels_complex_amplitude), Create(circ_complex_amplitude))
            # Here it is separated because the dot has to be created after the axes, or it glitches..
            scanning_dot_1.move_to(POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP)
            self.play(Create(line_complex_amplitude), Create(dot_complex_amplitude), Create(scanning_dot_1), run_time=2)
            complex_amplitude_ax_group = VGroup(ax_complex_amplitude, labels_complex_amplitude)
            complex_amplitude_graph_group = VGroup(complex_amplitude_ax_group, circ_complex_amplitude,
                                                   line_complex_amplitude, dot_complex_amplitude)
            # self.next_slide()
            self.wait_until_bookmark('A')
            self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(1), run_time=4)
            # self.play(TRACKER_SCANNING_SAMPLE.animate.increment_value(1), run_time=4)
            self.wait_until_bookmark('B')
            TRACKER_SCANNING_CAMERA.set_value(0)
            # self.next_slide()
            self.play(Create(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2)), run_time=tracker.time_until_bookmark('C'))
            # self.play(Create(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2)), run_time=2)
            constant_intensity_function = ax_2.plot(lambda x: 0.3, color=COLOR_INTENSITIES)
            camera_scanner_group.add(constant_intensity_function)
            camera_scanner_group.remove(amplitude_graph_2)
            # self.next_slide()
            self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(constant_intensity_function), run_time=max(tracker.get_remaining_duration() - 2, 1))
            # self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(constant_intensity_function), run_time=2)
            self.play(focus_arrow.animate.become(Arrow(start=ax_2.c2p(-0.05, 0.3) - np.array([np.sqrt(2)*0.9, 0, 0]), end=ax_2.c2p(-0.05, 0.3), color=RED)))
            problem_label = Tex(r'Problem!').scale(0.5).next_to(focus_arrow, LEFT)
            self.play(FadeIn(problem_label, shift=0.5*DOWN))
            # self.next_slide()
            self.updated_object_animation([microscope_VGroup, phase_image, camera_scanner_group, scanning_dot_1, focus_arrow, problem_label],
                                          FadeOut)
            TRACKER_SCANNING_CAMERA.set_value(0)
        # END INDENTATION
        # ################################################################################################################
        # # Magnified complex plane and presenting the perturbance approach:
        with self.voiceover(
                text="""In order to solve this problem, let's change our mathematical description of the outgoing wave.
                <bookmark mark='A'/> Instead of thinking of it as a complex number that slightly rotates in the complex
                plane, <bookmark mark='B'/> we can represent the same behavior as a constant unperturbed number to
                which we added a small almost perpendicular perturbation""") as tracker:
            self.play(complex_amplitude_ax_group.animate.move_to([0, 0, 0]).scale(scale_factor=2.5),
                      dot_complex_amplitude.animate.set_fill(COLOR_SCANNING_DOT),
                      FadeOut(title_1, shift=dy * UP),
                      title_2.animate.move_to([title_2.get_center()[0], y_0, 0]),
                      title_3.animate.move_to([title_3.get_center()[0], y_1, 0]),
                      FadeIn(title_4, shift=dy * UP),
                      titles_square.animate.set_width(title_3.width + 0.1).move_to([title_3.get_center()[0], y_1, 0])
                      )  # ,
            self.wait_until_bookmark('A')
            # self.next_slide()
            self.play(TRACKER_SCANNING_SAMPLE.animate.increment_value(1), run_time=4)
            self.wait(tracker.time_until_bookmark('B'))
            # self.next_slide()
            def line_amplitude_perturbation_generator():
                line_amplitude_perturbation = Line(
                    start=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0),
                    end=ax_complex_amplitude.c2p(
                        AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                            PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                        AMPLITUDE_SIZE * np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                            PHASE_OBJECT_SPATIAL_FREQUENCY * PI * TRACKER_SCANNING_SAMPLE.get_value()))),
                    color=COLOR_PERTURBED_AMPLITUDE)
                return line_amplitude_perturbation

            line_amplitude_perturbation = always_redraw(line_amplitude_perturbation_generator)
            line_complex_amplitude.clear_updaters()
            line_complex_amplitude.add_updater(lambda m: m.become(Line(start=ax_complex_amplitude.c2p(0, 0),
                                                                       end=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0),
                                                                       color=COLOR_UNPERTURBED_AMPLITUDE,
                                                                       z_index=ax_complex_amplitude.z_index + 1)),
                                               )
            tex = MathTex(r"\psi_{\text{out}}\left(x\right)="
                          r"\psi_{\text{unperturbed}}+i\psi_{\text{perturbation}}\left(x\right)").next_to(
                ax_complex_amplitude.get_bottom(), RIGHT + UP).scale(0.6)
            tex[0][8:20].set_color(COLOR_UNPERTURBED_AMPLITUDE)
            tex[0][22:].set_color(COLOR_PERTURBED_AMPLITUDE)
            self.play(FadeIn(tex), FadeIn(line_amplitude_perturbation))
            # self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(2), run_time=max(4, tracker.get_remaining_duration()))
            self.play(TRACKER_SCANNING_SAMPLE.animate.increment_value(1), run_time=4)
        # # END INDENTATION
        self.wait(0.3)
        # self.next_slide()

        with self.voiceover(
                text="""Just as we decomposed the amplitude to a constant unperturbed complex number and a perturbation,
                we can decompose the whole outgoing wave into the unperturbed wave plane, which is drawn here in orange,
                and other perturbation plane waves that are drawn in blue.
                <bookmark mark='A'/> Note how in between the lenses, the different components of the wave are spatially
                separated. We can take advantage of this spatial separation to manipulate only the orange unperturbed
                wave.""") as tracker:
            self.play(complex_amplitude_ax_group.animate.move_to(POSITION_AXES_1).scale(scale_factor=1 / 2.5),
                      FadeOut(tex))
            line_complex_amplitude.clear_updaters()
            # Fourier decomposition of the outgoing wave:
            # self.next_slide()
            self.play(FadeIn(microscope_VGroup), FadeIn(phase_image), FadeOut(complex_amplitude_graph_group))

            sample_outgoing_unperturbed_waves = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                                      end_point=POSITION_LENS_1,
                                                                                      wavelength=WAVELENGTH,
                                                                                      width=HEIGHT_SAMPLE,
                                                                                      tracker=TRACKER_TIME,
                                                                                      colors_generator=lambda
                                                                                          t: COLOR_UNPERTURBED_AMPLITUDE)
            sample_outgoing_perturbed_waves_1 = generate_wavefronts_start_to_end_flat(
                start_point=POSITION_SAMPLE + 0.2 * UP,
                end_point=POSITION_LENS_1 - 0.2 * UP,
                wavelength=WAVELENGTH,
                width=HEIGHT_SAMPLE,
                tracker=TRACKER_TIME,
                colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
            sample_outgoing_perturbed_waves_2 = generate_wavefronts_start_to_end_flat(
                start_point=POSITION_SAMPLE - 0.2 * UP,
                end_point=POSITION_LENS_1 + 0.2 * UP,
                wavelength=WAVELENGTH,
                width=HEIGHT_SAMPLE,
                tracker=TRACKER_TIME,
                colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

            gaussian_beam_waves_unperturbed = generate_wavefronts_start_to_end_gaussian(
                start_point=POSITION_LENS_1,
                end_point=POSITION_LENS_2,
                tracker=TRACKER_TIME,
                wavelength=WAVELENGTH,
                x_R=X_R,
                w_0=W_0,
                center=POSITION_WAIST,
                colors_generator=lambda
                    t: COLOR_UNPERTURBED_AMPLITUDE)
            gaussian_beam_waves_perturbed_1 = generate_wavefronts_start_to_end_gaussian(
                start_point=POSITION_LENS_1 + W_0 * UP,
                end_point=POSITION_LENS_2 + 4 * W_0 * UP,
                tracker=TRACKER_TIME,
                wavelength=WAVELENGTH,
                x_R=X_R,
                w_0=W_0,
                center=POSITION_WAIST + 2.3 * W_0 * UP,
                colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
            gaussian_beam_waves_perturbed_2 = generate_wavefronts_start_to_end_gaussian(
                start_point=POSITION_LENS_1 - W_0 * UP,
                end_point=POSITION_LENS_2 - 4 * W_0 * UP,
                tracker=TRACKER_TIME,
                wavelength=WAVELENGTH,
                x_R=X_R,
                w_0=W_0,
                center=POSITION_WAIST - 2.3 * W_0 * UP,
                colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

            second_lens_outgoing_waves_unperturbed = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                           end_point=POSITION_CAMERA,
                                                                                           wavelength=WAVELENGTH,
                                                                                           width=HEIGHT_CAMERA,
                                                                                           tracker=TRACKER_TIME,
                                                                                           colors_generator=lambda
                                                                                               t: COLOR_UNPERTURBED_AMPLITUDE)
            second_lens_outgoing_waves_purterbed_1 = generate_wavefronts_start_to_end_flat(
                start_point=POSITION_LENS_2 - 0.4 * UP,
                end_point=POSITION_CAMERA,
                wavelength=WAVELENGTH,
                width=HEIGHT_CAMERA,
                tracker=TRACKER_TIME,
                colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
            second_lens_outgoing_waves_purterbed_2 = generate_wavefronts_start_to_end_flat(
                start_point=POSITION_LENS_2 + 0.4 * UP,
                end_point=POSITION_CAMERA,
                wavelength=WAVELENGTH,
                width=HEIGHT_CAMERA,
                tracker=TRACKER_TIME,
                colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
            # self.next_slide()
            self.updated_object_animation([sample_outgoing_waves_moises,
                                           gaussian_beam_waves_moises,
                                           second_lens_outgoing_waves_moises], FadeOut)
            focus_arrow = create_focus_arrow_object(point=POSITION_WAIST + 0.1*UP - 0.07*RIGHT)
            self.updated_object_animation([sample_outgoing_unperturbed_waves,
                                           sample_outgoing_perturbed_waves_1,
                                           sample_outgoing_perturbed_waves_2,
                                           gaussian_beam_waves_unperturbed,
                                           gaussian_beam_waves_perturbed_1,
                                           gaussian_beam_waves_perturbed_2,
                                           second_lens_outgoing_waves_unperturbed,
                                           second_lens_outgoing_waves_purterbed_1,
                                           second_lens_outgoing_waves_purterbed_2
                                           ], FadeIn)
            # self.next_slide()
            self.play(TRACKER_TIME.animate.increment_value(tracker.time_until_bookmark('A')), run_time=tracker.time_until_bookmark('A'), rate_func=linear)
            # self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
            self.play(TRACKER_TIME.animate.increment_value(2), FadeIn(focus_arrow, shift=0.2*DOWN), run_time=2, rate_func=linear)
            # self.next_slide(loop=True)
            self.play(TRACKER_TIME.animate.increment_value(tracker.get_remaining_duration()), run_time=tracker.get_remaining_duration(), rate_func=linear)
            # self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
            # self.next_slide()
        # # END INDENTATION

        # ################################################################################################################
        # # Add laser:
        # laser_waves = generate_wavefronts_start_to_end_gaussian(
        #     start_point=POSITION_WAIST + LENGTH_LASER_BEAM * UP,
        #     end_point=POSITION_WAIST - LENGTH_LASER_BEAM * UP,
        #     tracker=TRACKER_TIME_LASER,
        #     wavelength=WAVELENGTH_LASER,
        #     x_R=X_R_LASER,
        #     w_0=W_0_LASER,
        #     center=POSITION_WAIST,
        #     colors_generator=lambda t: RED)
        laser_waves = Dot(point=POSITION_WAIST, radius=0.1, color=RED, fill_opacity=0.8, stroke_opacity=1, stroke_width=4, stroke_color=RED_E)

        orange_rgb = color_to_rgb(COLOR_PHASE_SHIFT_AMPLITUDE)
        white_rgb = color_to_rgb(COLOR_UNPERTURBED_AMPLITUDE)
        phase_shift_color_generator = lambda x: white_rgb * (1 - sigmoid(x)) + orange_rgb * sigmoid(x)
        gaussian_beam_waves_phase_shifted = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_LENS_1,
            end_point=POSITION_LENS_2,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST,
            colors_generator=phase_shift_color_generator)

        second_lens_outgoing_waves_shifted = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                   end_point=POSITION_CAMERA,
                                                                                   wavelength=WAVELENGTH,
                                                                                   width=HEIGHT_CAMERA,
                                                                                   tracker=TRACKER_TIME,
                                                                                   colors_generator=phase_shift_color_generator)
        with self.voiceover(
                text="""It is now time to introduce our solution to the problem: <bookmark mark='A'/> The laser phase plate.
                """) as tracker:
            self.play(FadeOut(title_2, shift=dy * UP),
                      title_3.animate.move_to([title_3.get_center()[0], y_0, 0]),
                      title_4.animate.move_to([title_4.get_center()[0], y_1, 0]),
                      FadeIn(title_5, shift=dy * UP),
                      titles_square.animate.set_width(title_4.width + 0.1).move_to([title_4.get_center()[0], y_1, 0])
                      )
            # self.next_slide()
            self.wait_until_bookmark('A')
            self.play(FadeOut(focus_arrow))
            self.updated_object_animation(laser_waves, FadeIn)
            self.wait(0.3)
        # # END INDENTATION
        # with self.voiceover(
        #         text="""By placing an intense laser right at the focal point of the lens, we can shine laser almost purely
        #         on the orange unperturbed wave. electron component that goes through a laser experience an effective higher potential energy,
        #         and therefore travels slower then components of the electron wave that pass by the laser. This adds
        #         relative phase shift to the unperturbed wave. the phase-shifted unperturbed wave is drawn in purple.""") as tracker:
        #     self.play(gaussian_beam_waves_unperturbed.animate.become(gaussian_beam_waves_phase_shifted),
        #               second_lens_outgoing_waves_unperturbed.animate.become(second_lens_outgoing_waves_shifted))
        #     self.remove(gaussian_beam_waves_unperturbed, second_lens_outgoing_waves_unperturbed)
        #     self.add(gaussian_beam_waves_phase_shifted, second_lens_outgoing_waves_shifted)
        #     # self.next_slide(loop=True)
        #     self.play(TRACKER_TIME.animate.increment_value(tracker.get_remaining_duration()), run_time=tracker.get_remaining_duration(), rate_func=linear)
        #     # self.next_slide()
        # # # END INDENTATION
        #
        #
        # left_side_group = VGroup(incoming_waves, sample, sample_outgoing_unperturbed_waves,
        #                          sample_outgoing_perturbed_waves_1, sample_outgoing_perturbed_waves_2,
        #                          gaussian_beam_waves_phase_shifted, gaussian_beam_waves_perturbed_1,
        #                          gaussian_beam_waves_perturbed_2, laser_waves, lens_1)
        # with self.voiceover(
        #         text="""Let's see how introducing the laser solves the problem""") as tracker:
        #     # self.next_slide()
        #     self.updated_object_animation([left_side_group, phase_image, sample_outgoing_waves_opacities, second_lens_outgoing_waves_opacities,
        #                 gaussian_beam_waves_opacities], FadeOut)
        # # # END INDENTATION
        # ################################################################################################################
        # # Complex plane recap:
        # complex_amplitude_graph_group.move_to(POSITION_LENS_1 + RIGHT).scale(2)
        # dot_complex_amplitude.scale(0.5)
        # TRACKER_SCANNING_CAMERA.set_value(0)
        # with self.voiceover(
        #         text="""Recall how the amplitude behaved on the camera before introducing the laser: It was just rotating in complex plane, right?""") as tracker:
        #     self.updated_object_animation([complex_amplitude_graph_group, scanning_dot_2], FadeIn)
        #     self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), TRACKER_SCANNING_SAMPLE.animate.set_value(1),
        #               run_time=tracker.get_remaining_duration())
        #     # self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), TRACKER_SCANNING_SAMPLE.animate.increment_value(1),
        #     #           run_time=4)
        # # # END INDENTATION
        # TRACKER_SCANNING_CAMERA.set_value(0), TRACKER_SCANNING_SAMPLE.set_value(0)
        # circ_complex_amplitude.clear_updaters()
        # dot_complex_amplitude.clear_updaters()
        # line_amplitude_perturbation.clear_updaters()
        # scanning_dot_2.move_to(POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP)
        # complex_amplitude_graph_group -= line_complex_amplitude
        # complex_amplitude_graph_group -= line_amplitude_perturbation
        # # self.next_slide()
        # # ################################################################################################################
        # # Rotate the unperturbed component:
        # with self.voiceover(
        #         text="""Well, now, since the laser induced a phase shift to the unperturbed component, the
        #         unperturbed component alone <bookmark mark='A'/> rotates in the complex plane. Specifically, we choose the intensity of the
        #         laser such that the rotation is exactly half pi. now - the unperturbed and perturbed components are
        #         parallel to first order!""") as tracker:
        #     self.wait_until_bookmark('A')
        #     self.play(line_complex_amplitude.animate.become(Line(start=ax_complex_amplitude.c2p(0, 0),
        #                                                          end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
        #                                                          color=COLOR_PHASE_SHIFT_AMPLITUDE,
        #                                                          z_index=ax_complex_amplitude.z_index + 1)),
        #           dot_complex_amplitude.animate.move_to(ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE)))
        #     # self.next_slide()
        # # # END INDENTATION
        # line_amplitude_perturbation = Line(start=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
        #                                    end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
        #                                    color=COLOR_PERTURBED_AMPLITUDE,
        #                                    z_index=line_complex_amplitude.z_index + 1)
        # complex_amplitude_graph_group += line_complex_amplitude
        # complex_amplitude_graph_group += line_amplitude_perturbation
        #
        # dot_complex_amplitude.add_updater(lambda m: m.move_to(
        #     ax_complex_amplitude.c2p(
        #         AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
        #             2 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
        #         AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
        #             2 * PI * TRACKER_SCANNING_CAMERA.get_value())) + 1))
        # ))
        # line_amplitude_perturbation.add_updater(lambda l: l.become(
        #     Line(start=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
        #          end=ax_complex_amplitude.c2p(
        #              AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
        #                  2 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
        #              AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
        #                  2 * PI * TRACKER_SCANNING_CAMERA.get_value())) + 1)),
        #          color=COLOR_PERTURBED_AMPLITUDE,
        #          z_index=ax_complex_amplitude.z_index + 1)))
        # with self.voiceover(
        #         text="""Now, when scanning the field across the camera's plane we see that the two components interfere
        #         constractively, and so the absolute value of the amplitude is no longer constant.""") as tracker:
        #     self.add(line_amplitude_perturbation)
        #     self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), run_time=tracker.get_remaining_duration())
        #     # self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), run_time=2)
        #     # self.next_slide()
        #     self.updated_object_animation([complex_amplitude_graph_group, scanning_dot_2], FadeOut)
        # # # END INDENTATION
        # TRACKER_SCANNING_CAMERA.set_value(0)
        # camera_scanner_group -= scanning_dot_2
        # camera_scanner_group.move_to(POSITION_LENS_1 - 0.2 * UP).scale(1.7)
        # phase_contrast_function = ax_2.plot(lambda x: 0.3 + 0.1 * np.sin(8 * np.pi * x) - 0.1 * np.cos(3 * np.pi * x),
        #                                     color=COLOR_INTENSITIES)
        # camera_scanner_group -= constant_intensity_function
        # scanning_dot_x_axis_2.scale(0.5)
        # scanning_dot_2.move_to(POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP)
        # with self.voiceover(
        #         text="""When the camera will measure the intensity - it will measure a photo which is no longer
        #         constant, and so we are finally able to see the phase object!""") as tracker:
        #     self.updated_object_animation([camera_scanner_group, scanning_dot_2], FadeIn)
        #     # self.next_slide()
        #     self.play(Create(phase_contrast_function), TRACKER_SCANNING_CAMERA.animate.set_value(1), run_time=tracker.get_remaining_duration())
        #     # self.play(Create(phase_contrast_function), TRACKER_SCANNING_CAMERA.animate.set_value(1), run_time=2)
        #     # self.next_slide()
        #     camera_scanner_group += phase_contrast_function
        #     self.updated_object_animation([camera_scanner_group, scanning_dot_2], FadeOut)
        #
        # # END INDENTATION
        # self.updated_object_animation(left_side_group, FadeIn)
        # # self.next_slide()
        #
        # Dt_e = 3/2
        # Dt_l = 1/2
        # alpha = np.arcsin(Dt_l * WAVELENGTH_LASER / (Dt_e * WAVELENGTH))
        # # ################################################################################################################
        # # Rotate the phase plate:
        # rotated_laser_waves = generate_wavefronts_start_to_end_gaussian(
        #     start_point=POSITION_WAIST + LENGTH_LASER_BEAM * UP * np.cos(alpha) - LENGTH_LASER_BEAM * RIGHT * np.sin(
        #         alpha),
        #     end_point=POSITION_WAIST - (
        #             LENGTH_LASER_BEAM * UP * np.cos(alpha) - LENGTH_LASER_BEAM * RIGHT * np.sin(alpha)),
        #     tracker=TRACKER_TIME_LASER,
        #     wavelength=WAVELENGTH_LASER,
        #     x_R=X_R_LASER,
        #     w_0=W_0_LASER,
        #     center=POSITION_WAIST,
        #     colors_generator=lambda t: RED)
        # with self.voiceover(
        #         text="""But can we do better? Due to camera's technology limitation, we can further improve the quality
        #         of the image by not only delaying the unperturbed component of the wave, but also to attenuate it and
        #         to darken the background of the sample.
        #         <bookmark mark='A'/> Luckily for us, there is an energy filter in the microscope that knows how to throw away electrons with
        #         different energies than the original electron energy. If we could only give or take some of the energy
        #         of the unperturbed wave, it would be filtered out later by the energy filter.
        #         The current set-up does not add or take energy from the electron, but only delays it.""") as tracker:
        #     self.play(FadeOut(title_3, shift=dy * UP),
        #               title_4.animate.move_to([title_4.get_center()[0], y_0, 0]),
        #               title_5.animate.move_to([title_5.get_center()[0], y_1, 0]),
        #               titles_square.animate.set_width(title_5.width + 0.1).move_to([title_5.get_center()[0], y_1, 0])
        #               )
        #     self.wait_until_bookmark("A")
        #     focus_arrow = create_focus_arrow_object(point=POSITION_ENERGY_FILTER + HEIGHT_CAMERA / 2 * UP)
        #     self.play(FadeIn(energy_filter, shift=DOWN), FadeIn(focus_arrow, shift=0.3*LEFT))
        #     microscope_VGroup += energy_filter
        #     self.play(Flash(energy_filter, color=RED, line_length=0.2, flash_radius=0.2))
        #     self.play(FadeOut(focus_arrow))
        #     self.play(TRACKER_TIME.animate.increment_value(tracker.get_remaining_duration()),
        #               run_time=tracker.get_remaining_duration(), rate_func=linear)
        # # # END INDENTATION
        # with self.voiceover(
        #         text="""Let's see what happens when we rotate the laser slightly and put two different wavelengths of light.
        #         The red lines of the laser represent nodes of high intensity.""") as tracker:
        #     # self.next_slide()
        #     self.play(laser_waves.animate.become(rotated_laser_waves), run_time=2)
        #     # self.next_slide()
        #     self.remove(laser_waves)
        #     self.add(rotated_laser_waves)
        # # # END INDENTATION
        # with self.voiceover(
        #         text="""Since there are now two different wavelengths,
        #                 the intensity beats, and the intensity nodes propagate in space.""") as tracker:
        #
        #     # self.next_slide(loop=True)
        #     self.play(TRACKER_TIME.animate.increment_value(Dt_e*tracker.get_remaining_duration()),
        #               TRACKER_TIME_LASER.animate.increment_value(Dt_l*tracker.get_remaining_duration()),
        #               run_time=tracker.get_remaining_duration(), rate_func=linear)
        #     # self.play(TRACKER_TIME.animate.increment_value(Dt_e * 2),
        #     #           TRACKER_TIME_LASER.animate.increment_value(Dt_l * 2),
        #     #           run_time=4, rate_func=linear)
        #     # self.next_slide()
        # # END INDENTATION
        #
        # # ################################################################################################################
        # # # Zoom in:
        # with self.voiceover(
        #         text="""Looking closely at the intersection point, we see that one can choose the angle of the
        #         laser tilt such that the electron's wavefronts surf on the intensity nodes. Each electron's wavefront
        #         experiences a constant intensity - which is different intensity than that of the following wavefront""") as tracker:
        #     self.play(self.camera.frame.animate.scale(ZOOM_RATIO).move_to(POSITION_WAIST - 0.4 * RIGHT))
        #     # self.next_slide(loop=True)
        #     self.play(TRACKER_TIME.animate.increment_value(Dt_e*tracker.get_remaining_duration()),
        #               TRACKER_TIME_LASER.animate.increment_value(Dt_l*tracker.get_remaining_duration()), run_time=tracker.get_remaining_duration(), rate_func=linear)
        #     # self.play(TRACKER_TIME.animate.increment_value(Dt_e * 2),
        #     #           TRACKER_TIME_LASER.animate.increment_value(Dt_l * 2), run_time=4, rate_func=linear)
        #     # self.next_slide()
        # # # END INDENTATION
        #
        #
        # self.updated_object_animation([lens_1, sample_outgoing_unperturbed_waves,
        #                                sample_outgoing_perturbed_waves_1, sample_outgoing_perturbed_waves_2], FadeOut)
        # complex_amplitude_graph_group.scale(ZOOM_RATIO).move_to(
        #     self.camera.frame_center - self.camera.frame_width / 4 * RIGHT + ZOOM_RATIO * 0.3 * UP)
        # lines_original_width = line_complex_amplitude.stroke_width
        # line_complex_amplitude.set_stroke(width=lines_original_width * ZOOM_RATIO)
        # line_amplitude_perturbation.set_stroke(width=lines_original_width * ZOOM_RATIO)
        # circ_complex_amplitude.set_stroke(width=lines_original_width * ZOOM_RATIO)
        # ax_complex_amplitude.set_stroke(width=lines_original_width * ZOOM_RATIO)
        # dot_complex_amplitude.scale(1)
        # graph_background = Rectangle(width=complex_amplitude_graph_group.width + 0.1,
        #                              height=complex_amplitude_graph_group.height + 0.1,
        #                              fill_opacity=1, fill_color=BLACK, stroke_width=0.2).move_to(
        #     complex_amplitude_graph_group.get_center())
        # waist_scanning_dot = Dot(color=COLOR_SCANNING_DOT, radius=0.05).move_to(POSITION_WAIST)
        # with self.voiceover(
        #     text="""Let's see how the beating affect the phase shift to the unperturbed wave."""):
        #     # self.next_slide()
        #     self.updated_object_animation([complex_amplitude_graph_group, graph_background, waist_scanning_dot], FadeIn)
        # # # END INDENTATION
        # single_frequency_laser_tex = Tex(r"Constant laser: $\psi\rightarrow\psi\cdot e^{i\frac{\pi}{2}}$").scale(0.5 * ZOOM_RATIO)
        # double_frequency_laser_tex = Tex(r"Beating laser: $\psi\rightarrow\psi\cdot e^{i\left(\frac{\pi}{2}+A\sin\left(\omega_{\text{beating}}t\right)\right)}$", r"$=e^{i\frac{\pi}{2}}\cdot\sum_{q\in\mathbb{Z}}a_{n}\cdot\psi\cdot e^{i\omega_{n}t}$").scale(0.5 * ZOOM_RATIO)
        # single_frequency_laser_tex[0][14].set_color(COLOR_UNPERTURBED_AMPLITUDE)
        # single_frequency_laser_tex[0][16:].set_color(COLOR_PHASE_SHIFT_AMPLITUDE)
        # double_frequency_laser_tex[0][13].set_color(COLOR_UNPERTURBED_AMPLITUDE)
        # double_frequency_laser_tex[0][16:].set_color(COLOR_PHASE_SHIFT_AMPLITUDE)
        # double_frequency_laser_tex[1][1:].set_color(COLOR_PHASE_SHIFT_AMPLITUDE)
        # single_frequency_laser_tex.next_to(ax_complex_amplitude, 0.3*DOWN).align_to(ax_complex_amplitude, LEFT)
        # double_frequency_laser_tex.next_to(single_frequency_laser_tex, 0.3*DOWN).align_to(single_frequency_laser_tex, LEFT)
        # with self.voiceover(
        #         text="""Earlier, we had a single frequency laser, which caused a constant phase shift to the unperturbed wave."""):
        #     # self.next_slide()
        #     self.play(FadeIn(single_frequency_laser_tex))
        # # # END INDENTATION
        # with self.voiceover(
        #         text="""Now, when the laser beats and each electron's wavefront experiences a different intensity, the
        #         phase shift itself is oscillating with time . We are left to show how this time dependent phase retardation, together with the energy
        #         filter will attenuate the intensity of the image.
        #         Since now the electron's wave has a phase delay which oscillates in time, we can use Fourier Transform to decompose it
        #         <bookmark mark='A'/> into a sum of different components, each one with a trivial well defined frequency.""") as tracker:
        #     self.play(FadeIn(double_frequency_laser_tex[0]))
        #
        #     modulation_rate = 2 * PI
        #     dot_complex_amplitude.add_updater(lambda m: m.move_to(
        #         ax_complex_amplitude.c2p(AMPLITUDE_SIZE * np.sin(-np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate)),
        #                                  AMPLITUDE_SIZE * np.cos(
        #                                      np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate)))))
        #     line_complex_amplitude.add_updater(lambda l: l.become(
        #         Line(start=ax_complex_amplitude.c2p(0, 0),
        #              end=ax_complex_amplitude.c2p(
        #                  AMPLITUDE_SIZE * np.sin(-np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate)),
        #                  AMPLITUDE_SIZE * np.cos(np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate))),
        #              stroke_width=lines_original_width * ZOOM_RATIO,
        #              color=COLOR_PHASE_SHIFT_AMPLITUDE,
        #              z_index=line_complex_amplitude.z_index + 1)))
        #     # self.next_slide(loop=True)
        #     self.play(TRACKER_TIME.animate.increment_value(Dt_e*tracker.time_until_bookmark('A')),
        #               TRACKER_TIME_LASER.animate.increment_value(Dt_l*tracker.time_until_bookmark('A')),
        #               run_time=tracker.time_until_bookmark('A'), rate_func=linear)
        #     # self.play(TRACKER_TIME.animate.increment_value(Dt_e*2),
        #     #           TRACKER_TIME_LASER.animate.increment_value(Dt_l*2),
        #     #           run_time=4, rate_func=linear)
        #     # self.next_slide()
        #     self.play(FadeIn(double_frequency_laser_tex[1]))
        #     self.wait(2)
        #     # self.next_slide()
        # # # END INDENTATION
        #
        # self.updated_object_animation(self.mobjects, FadeOut)
        # ################################################################################################################
        # # # Zoom out:
        # complex_amplitude_graph_group.scale(1 / ZOOM_RATIO).move_to([-3.5, 0, 0])
        # dot_complex_amplitude.move_to(ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE))
        # line_complex_amplitude.become(
        #               Line(start=ax_complex_amplitude.c2p(0, 0),
        #                    end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
        #                    stroke_width=lines_original_width,
        #                    color=COLOR_PHASE_SHIFT_AMPLITUDE,
        #                    z_index=ax_complex_amplitude.z_index + 1))
        # line_amplitude_perturbation.set_stroke(width=lines_original_width)
        # circ_complex_amplitude.set_stroke(width=lines_original_width)
        # ax_complex_amplitude.set_stroke(width=lines_original_width)
        # # ################################################################################################################
        # # Spectral line visualization:
        # energy_spectrum_axes = Axes(x_range=[-1, 1, 0.25],
        #                             y_range=[-1, 1, 0.25],
        #                             x_length=5,
        #                             y_length=5,
        #                             tips=False).move_to([-complex_amplitude_graph_group.get_center()[0], 0, 0])
        #
        # labels_complex_amplitude = energy_spectrum_axes.get_axis_labels(
        #     Tex(r'$\omega,E$'), Tex(r'$\psi$'))
        #
        # DELTA_W = 0.2
        # spectral_lines_generators = [lambda n=n: Line(start=energy_spectrum_axes.c2p(DELTA_W * n, 0),
        #                                               end=energy_spectrum_axes.c2p(DELTA_W * n, special.jv(n,
        #                                                                                                    TRACKER_PHASE_MODULATION.get_value()) ** 2),
        #                                               color=PURPLE_D,
        #                                               stroke_width=5,
        #                                               z_index=energy_spectrum_axes.z_index+1) for n in range(-4, 4)]
        # spectral_lines = [always_redraw(spectral_lines_generator) for spectral_lines_generator in
        #                   spectral_lines_generators]
        # line_complex_amplitude.clear_updaters()
        # dot_complex_amplitude.clear_updaters()
        #
        # with self.voiceover(
        #         text="""Lets see this effect on the energy spectrum of the electron, together with it's total complex
        #                 amplitude. <bookmark mark='R'/> Before encountering the laser, all of the electron's wavefunction was concentrated
        #                  in a single energy.
        #                  <bookmark mark='A'/> Now, as we turn up the laser, it decomposes into a sum of many energies. <bookmark mark='K'/> As the energy filter will filter out all the energies which are not the original one, we will
        #                  be left with attenuated amplitude of the original unperturbed wave""") as tracker:
        #     self.play(self.camera.frame.animate.scale(1 / ZOOM_RATIO).move_to(ORIGIN))
        #     self.updated_object_animation([complex_amplitude_graph_group, energy_spectrum_axes, labels_complex_amplitude], FadeIn)
        #     self.updated_object_animation(spectral_lines, FadeIn)
        #     # self.next_slide()
        #     self.wait_until_bookmark('R')
        #     focus_arrow = Arrow(start=energy_spectrum_axes.c2p(0, special.jv(0, TRACKER_PHASE_MODULATION.get_value())) + [-0.9, 0.9, 0],
        #                         end=energy_spectrum_axes.c2p(0, special.jv(0, TRACKER_PHASE_MODULATION.get_value())),
        #                         color=RED)
        #     self.play(FadeIn(focus_arrow, shift=0.3*DOWN))
        #     # self.next_slide()
        #     focus_arrow.add_updater(lambda l: l.become(Arrow(
        #         start=energy_spectrum_axes.c2p(0, special.jv(0, TRACKER_PHASE_MODULATION.get_value()) ** 2) + [-0.9, 0.9, 0],
        #         end=energy_spectrum_axes.c2p(0, special.jv(0, TRACKER_PHASE_MODULATION.get_value()) ** 2),
        #         color=RED)))
        #     self.wait_until_bookmark('A')
        #     self.play(TRACKER_PHASE_MODULATION.animate.increment_value(2), run_time=5)
        #     line_complex_amplitude.add_updater(lambda l: l.become(
        #         Line(start=ax_complex_amplitude.c2p(0, 0),
        #              end=ax_complex_amplitude.c2p(0,
        #                                           AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION_SECONDARY.get_value())),
        #              stroke_width=lines_original_width,
        #              color=PURPLE_B,
        #              z_index=ax_complex_amplitude.z_index + 1).set_color(PURPLE_B)))  # I am not sure why he does not make him purple_b without this extra explicit command
        #
        #     dot_complex_amplitude.add_updater(lambda l: l.move_to(
        #         ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION_SECONDARY.get_value()))))
        #     self.wait_until_bookmark('K')
        #     focus_arrow.clear_updaters()
        #     self.play(focus_arrow.animate.move_to(
        #         ax_complex_amplitude.c2p(
        #             0,
        #             AMPLITUDE_SIZE * special.jv(
        #                 0,
        #                 TRACKER_PHASE_MODULATION_SECONDARY.get_value())) + np.array([-0.5, 0.5, 0])))
        #     focus_arrow.add_updater(lambda l: l.move_to(ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE * special.jv(0,
        #                                                                                                         TRACKER_PHASE_MODULATION_SECONDARY.get_value())) + np.array(
        #         [-0.5, 0.5, 0])))
        #     self.updated_object_animation(list(set(spectral_lines).difference(spectral_lines[4])),
        #                                   lambda m: m.animate.set_color(GRAY))
        #     [spectral_line.clear_updaters() for spectral_line in spectral_lines]
        #     self.play(TRACKER_PHASE_MODULATION_SECONDARY.animate.increment_value(2), run_time=5)
        #     # # END INDENTATION
        #
        #
        #
        # # self.next_slide()
        # line_amplitude_perturbation.clear_updaters()
        # line_amplitude_perturbation.add_updater(lambda l: l.become(
        #                             Line(start=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value())),
        #                                  end=ax_complex_amplitude.c2p(
        #                                                       AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
        #                                                           6 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
        #                                                       AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value()) +
        #                                                       AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
        #                                                           6 * PI * TRACKER_SCANNING_CAMERA.get_value())))),
        #                                  color=COLOR_PERTURBED_AMPLITUDE,
        #                                  z_index=line_complex_amplitude.z_index + 1
        #                                  ),
        #
        # ))
        # dot_complex_amplitude.clear_updaters()
        # dot_complex_amplitude.add_updater(lambda l: l.move_to(ax_complex_amplitude.c2p(
        #                                                       AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
        #                                                           6 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
        #                                                       AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value()) +
        #                                                       AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
        #                                                           6 * PI * TRACKER_SCANNING_CAMERA.get_value()))))))
        # with self.voiceover(
        #         text="""Now, when the camera will measure the field's intensity, There will be no bright background to
        #         the object, and the image will contain only the signal itself.""") as tracker:
        #     self.play(TRACKER_SCANNING_CAMERA.animate.increment_value(1), run_time=5)
        # # # END INDENTATION
        #
        # # self.next_slide()
        # with self.voiceover(
        #         text="""Thank you for watching! To hear more about our work, search 'Osip Schwarz lab' in google.""") as tracker:
        #     self.updated_object_animation(self.mobjects, FadeOut)
        #
        #     # final_title = Tex("Questions?", color=WHITE).scale(1.5)
        #     final_title = Tex("To hear more search 'Osip Schwarz lab' in Google.", color=WHITE).scale(0.8)
        #     self.play(Write(final_title))
        #     # self.next_slide()
        #     # self.wait(tracker.get_remaining_duration())
        #     self.wait(1)
        # # # END INDENTATION
        # self.play(FadeOut(final_title))

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


from manim import *
import numpy as np
import matplotlib.pyplot as plt

BACKGROUND_COLOR = ORANGE  # background color behind SVG shape
SAMPLE_COLOR = PURPLE      # SVG shape fill color
SVG_PATH = r"phd-project/sea horse.svg"


from manim import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys

BACKGROUND_COLOR = "#29505B"
TEXT_COLOR = "#D8E1E3"
BACKGROUND_NOISE_PATH = "phd-project/hue_noise_background.png"
SAMPLE_COLOR = PURPLE
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
        h = np.clip(np.random.normal(loc=0.3, scale=0.1, size=(100, 100)), 0, 1)
        s = np.full_like(h, SATURATION)
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

        # all_mobjects = Group(
        #     intensity_image, hue_noise_image, phase_image, left_axes, right_axes, left_title, right_title, sup_title)



COLOR_ARC = BLUE
COLOR_CIRCLE = GRAY
COLOR_LINE = GOLD_B
COLOR_ARROW = TEAL
WIDTH_LINES_THICK = 3.5
WIDTH_LINES_THIN = 1.5


class ParametricArcScene(Scene):
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
        label_signal = Tex("signal", font_size=35, color=TEXT_COLOR).next_to(legend_signal, RIGHT)
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


COLOR_LENS = BLUE
COLOR_MIRROR = GREEN
COLOR_LASER = RED
COLOR_ELECTRON = PURPLE

class Cavity(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        mm = 0.1
        global_shift = -6
        global_vertical_shift = -1

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
        big_mirror_location = 120*mm
        big_mirror_D = 25.4*mm
        big_mirror_arc_angle = np.arcsin(big_mirror_D / (2 * big_mirror_R))

        big_mirror_arc = Arc(
            radius=big_mirror_R,
            start_angle=-big_mirror_arc_angle,
            angle=2 * big_mirror_arc_angle,
            color=COLOR_MIRROR,
            stroke_width=WIDTH_LINES_THICK
        ).move_to([big_mirror_location + global_shift, global_vertical_shift, 0])
        a = ValueTracker(0.6)
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
                                                               x_R=X_R*5,
                                                               w_0=W_0/1.5,
                                                               center=None,
                                                               colors_generator=lambda t: COLOR_LASER)
        waves_lens = generate_wavefronts_start_to_end_gaussian(start_point=[lens_left_location + global_shift, global_vertical_shift, 0],
                                                                end_point=[lens_right_location + global_shift, global_vertical_shift, 0],
                                                                tracker=c,
                                                                wavelength=WAVELENGTH,
                                                                x_R=X_R * 2,
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

        legend_start = [-1.3, 3+global_vertical_shift, 0]  # Top-right relative position

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






# m = Microscope()
# m.construct()

# manim -pql slides/scene.py Microscope
# manim-slides convert Microscope slides/presentation.html
