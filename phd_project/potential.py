from manim import *


ARCS_RADIUS = 3
THETA_P_1 = PI / 8
MIRRORS_NA = PI / 2
KERNEL_QUADRATIC_COEFFICIENT = 50
unconcentricity=0.01
RIGHT_ARC_CENTER = np.array([-unconcentricity, 0, 0])
MIRROR_2_CENTER = ORIGIN

ALGEBRAIC_EXPRESSIONS_SCALE = 0.8

SCANNING_DOT_TRACKER = ValueTracker(PI - MIRRORS_NA / 2)
SCANNING_DOT_RADIUS_TRACKER = ValueTracker(0.08)

COLOR_MIRRORS = WHITE
COLOR_INTEGRAL = ORANGE
COLOR_P_1 = RED
DISTANCES_COLOR = BLUE
from scipy.special import fresnel


def frensel_ax2_integral(a, x_0, x_1):
    factor = np.sqrt(np.pi / (2 * a))
    u0 = np.sqrt(2 * a / np.pi) * x_0
    u1 = np.sqrt(2 * a / np.pi) * x_1
    v0 = np.array(fresnel(u0))
    v1 = np.array(fresnel(u1))
    return factor * (v1 - v0)


class Potential(ZoomedScene):
    def construct(self):
        # TODO: choose the real point which is on mirror_1 and at angle theta_1
        # Basic system generation:
        p_1 = RIGHT_ARC_CENTER + ARCS_RADIUS * np.cos(THETA_P_1) * RIGHT + ARCS_RADIUS * np.sin(THETA_P_1) * UP
        p_1_reflection = ARCS_RADIUS * np.cos(THETA_P_1 + PI) * RIGHT + ARCS_RADIUS * np.sin(THETA_P_1 + PI) * UP
        mirror_1 = Arc(arc_center=RIGHT_ARC_CENTER, start_angle=-MIRRORS_NA / 2, angle=MIRRORS_NA, radius=ARCS_RADIUS, color=COLOR_MIRRORS)
        mirror_2 = Arc(start_angle=PI - MIRRORS_NA / 2, angle=MIRRORS_NA, radius=ARCS_RADIUS, color=COLOR_MIRRORS)
        p_1_dot = Dot(color=COLOR_P_1, point=p_1)
        p_1_reflection_dot = always_redraw(lambda: DashedVMobject(Dot(color=COLOR_P_1, point=p_1_reflection, radius=SCANNING_DOT_RADIUS_TRACKER.get_value(), stroke_width=1, fill_opacity=0)))
        p_0_dot = always_redraw(
            lambda: Dot(
                color=RED,
                radius=SCANNING_DOT_RADIUS_TRACKER.get_value(),
                point=ARCS_RADIUS * np.cos(SCANNING_DOT_TRACKER.get_value()) * RIGHT
                      + ARCS_RADIUS * np.sin(SCANNING_DOT_TRACKER.get_value()) * UP,
            )
        )
        p_0_to_p_1_line = always_redraw(lambda: Line(p_0_dot.get_center(), p_1, color=WHITE))
        line_length_label = always_redraw(lambda: Tex(f"$r_{{12}}={np.linalg.norm(p_0_dot.get_center() - p_1):.2f}$").next_to(p_0_to_p_1_line.get_center(), UP, buff=0.1).rotate(np.arctan2(*(p_1 - p_0_dot.get_center())[[1, 0]])))
        p_1_label = Tex(r"$p_{1}$").next_to(p_1, UR, buff=0.1)
        p_0_label = always_redraw(lambda: Tex(r"$p_{0}$").next_to(p_0_dot, LEFT, buff=0.1))


        # Integrand and integral representations' generation:
        box_integrand = self.SmallAxesBox(2.5).to_corner(DR)
        box_integral = self.SmallAxesBox(2.5).next_to(box_integrand, UP, buff=1)
        plane_integrand = box_integrand[0]  # extract the NumberPlane
        plane_integral = box_integral[0]  # extract the NumberPlane
        phase_representation = always_redraw(
            lambda: Line(
                plane_integrand.c2p(0, 0),
                plane_integrand.c2p(
                    np.cos(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value())),
                    np.sin(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value())),
                ),
                color=COLOR_INTEGRAL,
                stroke_width=1
            )
        )

        integral_representation = always_redraw(
            lambda: ParametricFunction(
                lambda t: plane_integral.c2p(*(6 * self.integral_curve(t))),  # unpack (x,y)
                t_range=(0.0, SCANNING_DOT_TRACKER.get_value()),
                color=COLOR_INTEGRAL,
                stroke_width=1
            ),
        )
        integrand_approximation = Tex(
            r"$r_{01}=r_{0}-\frac{r_{01,\max}-R}{2Rr_{01,\max}}s^{2}+\mathcal{O}\left(s_{0}^{4}\right)$").to_edge(LEFT).shift(UP)#next_to(
            #self.zoomed_display, UP, buff=0.1)

        # Distances helpers generation:
        p_1_circle_radius = np.linalg.norm(p_1 - p_1_reflection)
        p_1_circle = DashedVMobject(Circle(radius=p_1_circle_radius, color=DISTANCES_COLOR, stroke_width=0.5).move_to(p_1), num_dashes=200, dashed_ratio=0.7)
        radius_line_end_point = p_1_reflection # p_1 + p_1_circle_radius * np.array([np.cos(PI - 0.2), np.sin(PI - 0.2), 0])
        p_1_circle_radius_line = DashedLine(p_1, radius_line_end_point, color=DISTANCES_COLOR, stroke_width=0.5, dash_length=2 * PI * p_1_circle_radius / 200, dashed_ratio=0.7)
        p_1_circle_radius_label = Tex(r"$\left(r_{01,\max}\right)$", color=DISTANCES_COLOR).next_to(p_1_circle_radius_line.get_center(), DOWN, buff=0.3).rotate(np.arctan2(*(p_1 - radius_line_end_point)[[1, 0]]))
        distances_group = VGroup(p_1_circle, p_1_circle_radius_line, p_1_circle_radius_label, p_1_reflection_dot)

        # Integral result label generation:
        integral_algebraic_label = Tex(r"$\frac{ke^{ikr_{01,\max}}}{4\pi ir_{01,\max}}\intop_{S}U\left(\boldsymbol{r}_{0}\right)e^{-ik\frac{r_{01,\max}-R}{2R\cdot r_{01,\max}}s^{2}}dS$").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_algebraic_label.next_to(box_integral, UP, buff=0.1).to_edge(RIGHT)
        integral_algebraic_expression_as_convolution = Tex(r"$\frac{ke^{ikr_{01,\min}}}{4\pi ir_{01,\min}}\cdot\left[e^{-ik\frac{r_{01,\min}-R}{2R\cdot r_{01,\min}}s^{2}}\otimes U\left(\boldsymbol{p}_{0}\right)\right]$").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_algebraic_expression_as_convolution.next_to(box_integral, UP, buff=0.1).to_edge(RIGHT)
        integral_arrow_indicator = Arrow(integral_algebraic_label.get_bottom(), plane_integral.c2p(0, 0), color=COLOR_INTEGRAL)
        integral_result_group = VGroup(integral_algebraic_label, integral_arrow_indicator)

        # Huygens integral introduction
        self.play(Create(mirror_1), Create(mirror_2))
        self.play(Create(p_1_dot), Create(p_0_to_p_1_line), Create(p_0_dot), FadeIn(line_length_label), FadeIn(p_1_label), FadeIn(p_0_label))
        self.add(mirror_1, mirror_2, p_1_dot, p_0_dot, p_0_to_p_1_line, box_integrand, box_integral, phase_representation, integral_representation, line_length_label)
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=2, rate_func=linear)

        # Zoomed display and move it to the right place generation:
        self.activate_zooming()
        self.zoomed_display.move_to(p_1_reflection)
        zf = self.zoomed_camera.frame.move_to(p_1_reflection)
        zf.set_width(1.0)

        # Repeat with zoom animation:
        self.play(Create(distances_group), mirror_1.animate.set_stroke(width=0.5), mirror_2.animate.set_stroke(width=0.5), p_1_dot.animate.scale(0.5), SCANNING_DOT_RADIUS_TRACKER.animate.set_value(0.04))
        self.play(SCANNING_DOT_TRACKER.animate.set_value(THETA_P_1 + PI - 0.19), run_time=1.0)
        self.play(Write(integrand_approximation), run_time=1)
        self.play(SCANNING_DOT_TRACKER.animate.set_value(THETA_P_1 + PI + 0.19), run_time=2, rate_func=linear)

        # Interpret algebraic expressions animation:
        self.play(FadeIn(integral_result_group), run_time=2)
        self.play(FadeOut(integral_algebraic_label), FadeIn(integral_algebraic_expression_as_convolution), run_time=2)
        # self.play(Uncreate())

    @staticmethod
    def integrand_phase_representation(theta) -> float:
        return KERNEL_QUADRATIC_COEFFICIENT * (theta - (THETA_P_1 + PI)) ** 2

    @staticmethod
    def integral_curve(theta) -> np.ndarray:
        return frensel_ax2_integral(
            a=KERNEL_QUADRATIC_COEFFICIENT, x_0=-(THETA_P_1 + MIRRORS_NA / 2), x_1=theta - (THETA_P_1 + PI)
        )

    @staticmethod
    def SmallAxesBox(
        side_length=3,
        x_range=(-2, 2, 1),
        y_range=(-2, 2, 1),
    ) -> VGroup:
        plane = NumberPlane(
            x_range=x_range,
            y_range=y_range,
            x_length=side_length,
            y_length=side_length,
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "stroke_width": 2,
            },
        )

        border = SurroundingRectangle(plane, buff=0)

        return VGroup(plane, border)

# Command to run the scene:
# manim -pql potential.py Potential
