from manim import *
from scipy.special import fresnel

MIRRORS_RADIUS = 3
MIRRORS_NA = PI / 2
KERNEL_QUADRATIC_COEFFICIENT = 50
UNCONCENTRICITY=0.7
ZOOMED_ANGLE_RANGE = 0.19

MIRROR_LEFT_CENTER = np.array([-1, 0, 0])
MIRROR_RIGHT_CENTER = MIRROR_LEFT_CENTER - np.array([UNCONCENTRICITY, 0, 0])

ALGEBRAIC_EXPRESSIONS_SCALE = 0.8

SCANNING_DOT_TRACKER = ValueTracker(PI - MIRRORS_NA / 2)
SCANNING_DOT_RADIUS_TRACKER = ValueTracker(0.08)
THETA_P_1_TRACKER = ValueTracker(PI / 8)

COLOR_MIRRORS = WHITE
COLOR_INTEGRAL = ORANGE
COLOR_P_1 = RED
DISTANCES_COLOR = BLUE


class Potential(ZoomedScene):
    def construct(self):
        # TODO: choose the real point which is on mirror_right and at angle theta_1
        # Basic system generation:
        p_1 = MIRROR_RIGHT_CENTER + MIRRORS_RADIUS * np.cos(THETA_P_1_TRACKER.get_value()) * RIGHT + MIRRORS_RADIUS * np.sin(THETA_P_1_TRACKER.get_value()) * UP
        p_1_prime = MIRROR_LEFT_CENTER + MIRRORS_RADIUS * np.cos(THETA_P_1_TRACKER.get_value() + PI) * RIGHT + MIRRORS_RADIUS * np.sin(THETA_P_1_TRACKER.get_value() + PI) * UP
        mirror_right = Arc(arc_center=MIRROR_RIGHT_CENTER, start_angle=-MIRRORS_NA / 2, angle=MIRRORS_NA, radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        mirror_left = Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=PI - MIRRORS_NA / 2, angle=MIRRORS_NA, radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        p_1_dot = Dot(color=COLOR_P_1, point=p_1)
        p_1_prime_dot = always_redraw(lambda: DashedVMobject(Dot(color=COLOR_P_1, point=p_1_prime, radius=SCANNING_DOT_RADIUS_TRACKER.get_value(), stroke_width=1, fill_opacity=0)))
        p_0_dot = always_redraw(
            lambda: Dot(
                color=RED,
                radius=SCANNING_DOT_RADIUS_TRACKER.get_value(),
                point=MIRROR_LEFT_CENTER + MIRRORS_RADIUS * np.cos(SCANNING_DOT_TRACKER.get_value()) * RIGHT
                      + MIRRORS_RADIUS * np.sin(SCANNING_DOT_TRACKER.get_value()) * UP,
            )
        )
        p_0_to_p_1_line = always_redraw(lambda: Line(p_0_dot.get_center(), p_1, color=WHITE))
        line_length_label = always_redraw(lambda: Tex(f"$r_{{01}}={np.linalg.norm(p_0_dot.get_center() - p_1):.2f}$").next_to(p_0_to_p_1_line.get_center(), UP, buff=0.1).rotate(np.arctan2(*(p_1 - p_0_dot.get_center())[[1, 0]])))
        p_1_label = Tex(r"$p_{1}$").next_to(p_1, UR, buff=0.1)
        p_1_prime_label = Tex(r"$p^{\prime}_{1}$").next_to(p_1_prime, DL, buff=0.0).scale(0.3)
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
                    np.cos(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value(), THETA_P_1_TRACKER.get_value())),
                    np.sin(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value(), THETA_P_1_TRACKER.get_value())),
                ),
                color=COLOR_INTEGRAL,
                stroke_width=1
            )
        )

        integral_representation = always_redraw(
            lambda: ParametricFunction(
                lambda t: plane_integral.c2p(*(self.integral_curve(t, THETA_P_1_TRACKER.get_value()))),  # unpack (x,y)
                t_range=(0.0, SCANNING_DOT_TRACKER.get_value()),
                color=COLOR_INTEGRAL,
                stroke_width=1
            ),
        )
        r_01_approximation = Tex(
            r"$r_{01}\approx r_{11^{\prime}}-\frac{r_{11^{\prime}}-R}{2Rr_{11^{\prime}}}\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2}$").to_edge(LEFT).shift(1.5*UP).scale(ALGEBRAIC_EXPRESSIONS_SCALE)#next_to( # This somewhy doesn't work.
            #self.zoomed_display, UP, buff=0.1)

        # p_1 Distances helpers generation:
        p_1_circle_radius = np.linalg.norm(p_1 - p_1_prime)
        p_1_circle = DashedVMobject(Circle(arc_center=p_1, radius=p_1_circle_radius, color=DISTANCES_COLOR, stroke_width=0.5), num_dashes=200, dashed_ratio=0.7)
        radius_line_end_point = p_1_prime # p_1 + p_1_circle_radius * np.array([np.cos(PI - 0.2), np.sin(PI - 0.2), 0])
        p_1_circle_radius_line = DashedLine(p_1, radius_line_end_point, color=DISTANCES_COLOR, stroke_width=0.5, dash_length=2 * PI * p_1_circle_radius / 200, dashed_ratio=0.7)
        p_1_circle_radius_label = Tex(r"$r_{11^{\prime}}$", color=DISTANCES_COLOR).next_to(p_1_circle_radius_line.get_center(), DOWN, buff=0.3).rotate(np.arctan2(*(p_1 - radius_line_end_point)[[1, 0]]))
        distances_group_p_1 = VGroup(p_1_circle, p_1_circle_radius_line, p_1_circle_radius_label, p_1_prime_dot, p_1_prime_label)
        r_11_prime_approximation_label = Tex(r"$r_{11^{\prime}}=2R-u\cos\left(\frac{p_{1}}{R}\right)+\mathcal{O}\left(\left(\frac{u}{R}\right)^{2}\right)$").to_edge(DOWN).scale(ALGEBRAIC_EXPRESSIONS_SCALE)

        # mirror_left Distances helpers generation:
        relevant_radius = MIRRORS_RADIUS - UNCONCENTRICITY
        mirror_left_circle = DashedVMobject(Circle(arc_center=MIRROR_LEFT_CENTER, radius=relevant_radius, color=DISTANCES_COLOR, stroke_width=0.5), num_dashes=200, dashed_ratio=0.7)
        mirror_left_radius_line = DashedLine(MIRROR_LEFT_CENTER, MIRROR_LEFT_CENTER + (MIRRORS_RADIUS - UNCONCENTRICITY) * RIGHT, color=DISTANCES_COLOR, stroke_width=0.5, dash_length=2 * PI * relevant_radius / 200, dashed_ratio=0.7)
        mirror_left_radius_label = Tex(r"$\min_{p_{1}}\left(r_{11^{\prime}}\right)$", color=DISTANCES_COLOR).scale(ALGEBRAIC_EXPRESSIONS_SCALE).next_to(mirror_left_radius_line.get_center(), DOWN, buff=0.3)
        distances_group_mirror_left = VGroup(mirror_left_circle, mirror_left_radius_line, mirror_left_radius_label)

        # Integral result label generation:
        integral_label = Tex(r"$U\left(\boldsymbol{p}_{1}\right)=\frac{ke^{ikr_{11^{\prime}}}}{4\pi ir_{11^{\prime}}}\intop_{S}U\left(\boldsymbol{p}_{0}\right)e^{-ik\frac{r_{11^{\prime}}-R}{2R\cdot r_{11^{\prime}}}\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2}}dS$").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_label.next_to(box_integral, UP, buff=0.1).to_edge(RIGHT)
        integral_expression_as_convolution = Tex(r"$U\left(\boldsymbol{p}_{1}\right)=\frac{ke^{ikr_{11^{\prime}}}}{4\pi ir_{11^{\prime}}}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-ik\frac{r_{11^{\prime}}-R}{2R\cdot r_{11^{\prime}}}\boldsymbol{p}_{0}^{2}}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)$").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_expression_as_convolution.next_to(box_integral, UP, buff=0.1).to_edge(RIGHT)
        integral_expression_substitute_r_11_prime = Tex(r"$=\frac{ke^{ik\left(2R-u\cos\left(\frac{s_{1}}{R}\right)\right)}}{8\pi iR}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-ik\frac{R}{2R_{0}\cdot}p_{0}^{2}}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)$").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_expression_with_separated_potential = Tex(r"$=\underset{\text{Constant phase}}{\underbrace{\frac{ke^{2iR}}{8\pi iR}}}\cdot\underset{\text{Position dependent phase}}{\underbrace{e^{-iku\cos\left(\frac{s_{1}}{R}\right)}}}\cdot\underset{\text{Convolution}}{\underbrace{\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-\frac{ik}{4R}p_{0}^{2}}\right]}}$").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_arrow_indicator = Arrow(integral_label.get_bottom(), plane_integral.c2p(*(self.integral_curve(THETA_P_1_TRACKER.get_value() + PI + ZOOMED_ANGLE_RANGE, THETA_P_1_TRACKER.get_value()))), color=COLOR_INTEGRAL)
        integral_result_group = VGroup(integral_label, integral_arrow_indicator)


        # Huygens integral introduction
        self.play(Create(mirror_right), Create(mirror_left))
        self.play(Create(p_1_dot), Create(p_0_to_p_1_line), Create(p_0_dot), FadeIn(line_length_label), FadeIn(p_1_label), FadeIn(p_0_label))
        self.add(mirror_right, mirror_left, p_1_dot, p_0_dot, p_0_to_p_1_line, box_integrand, box_integral, phase_representation, integral_representation, line_length_label)
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=1, rate_func=linear)

        # Zoomed display and move it to the right place generation:
        self.activate_zooming()
        zoomed_display = self.zoomed_display
        zoomed_display.move_to(p_1_prime)
        frame = self.zoomed_camera.frame
        frame.move_to(p_1_prime)
        frame.set_width(1.0)

        # Repeat with zoom animation:
        self.play(Create(distances_group_p_1), mirror_right.animate.set_stroke(width=0.5), mirror_left.animate.set_stroke(width=0.5), p_1_dot.animate.scale(0.5), SCANNING_DOT_RADIUS_TRACKER.animate.set_value(0.04))
        self.play(SCANNING_DOT_TRACKER.animate.set_value(THETA_P_1_TRACKER.get_value() + PI - ZOOMED_ANGLE_RANGE), run_time=1.0)
        self.play(Write(r_01_approximation), run_time=1)
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=1, rate_func=linear)

        # Interpret algebraic expressions animation:
        self.play(FadeIn(integral_result_group), run_time=2)
        self.play(FadeOut(integral_label, shift=UP), FadeOut(r_01_approximation),
                  FadeOut(p_0_to_p_1_line), FadeOut(line_length_label), FadeIn(integral_expression_as_convolution, shift=UP), run_time=2)

        # Change discussion to r_01:
        self.play(Uncreate(distances_group_p_1))
        self.play(Create(distances_group_mirror_left), run_time=2)
        self.play(zoomed_display.animate.move_to(MIRROR_RIGHT_CENTER + MIRRORS_RADIUS * RIGHT), frame.animate.move_to(MIRROR_RIGHT_CENTER + MIRRORS_RADIUS * RIGHT))
        self.play(FadeIn(r_11_prime_approximation_label), run_time=2)

        # Focus on the convolution and its interpretation animation:
        self.play(FadeOut(Group(mirror_right, mirror_left, p_1_dot, p_0_dot, box_integrand, box_integral,
                                phase_representation, integral_representation, distances_group_mirror_left,
                                integral_arrow_indicator, p_0_label, p_1_label)))
        self.play(FadeOut(frame), FadeOut(zoomed_display))
        self.play(integral_expression_as_convolution.animate.move_to(ORIGIN))
        self.play(integral_expression_as_convolution.animate.shift(1.5*UP),
                  FadeIn(integral_expression_substitute_r_11_prime, shift=UP))
        self.play(integral_expression_as_convolution.animate.shift(1.5*UP),
                  integral_expression_substitute_r_11_prime.animate.shift(1.5*UP),
                  FadeIn(integral_expression_with_separated_potential, shift=UP), run_time=2)
        self.wait(2)

    @staticmethod
    def integrand_phase_representation(theta, theta_p_1) -> float:
        return KERNEL_QUADRATIC_COEFFICIENT * (theta - (theta_p_1 + PI)) ** 2

    @staticmethod
    def integral_curve(theta, theta_p_1) -> np.ndarray:
        return 6 * frensel_ax2_integral(
            a=KERNEL_QUADRATIC_COEFFICIENT, x_0=-(theta_p_1 + MIRRORS_NA / 2), x_1=theta - (theta_p_1 + PI)
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
def frensel_ax2_integral(a, x_0, x_1):
    factor = np.sqrt(np.pi / (2 * a))
    u0 = np.sqrt(2 * a / np.pi) * x_0
    u1 = np.sqrt(2 * a / np.pi) * x_1
    v0 = np.array(fresnel(u0))
    v1 = np.array(fresnel(u1))
    return factor * (v1 - v0)


def find_intersection_with_ray(ray_origin, circle_origin, angle, circle_radius) -> np.ndarray:
    # The following expression is the result of calculation "Intersection of a parameterized line and a sphere"
    # in the research lyx file
    k_vector = np.array([np.cos(angle), np.sin(angle), 0])  # m_rays | 3
    Delta = ray_origin - circle_origin  # m_rays | 3
    Delta_squared = np.sum(Delta**2, axis=-1)  # m_rays
    Delta_projection_on_k = np.sum(Delta * k_vector, axis=-1)  # m_rays
    with np.errstate(invalid="ignore"):
        length = -Delta_projection_on_k + np.sqrt(
            Delta_projection_on_k**2 - Delta_squared + circle_radius**2
        )
    intersection_point = ray_origin + length * k_vector
    return intersection_point