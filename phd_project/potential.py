import numpy as np
from manim import *
from scipy.special import fresnel

tex_template = TexTemplate()
tex_template.add_to_preamble(r"\usepackage{dsfont}")

MIRRORS_RADIUS = 3
MIRRORS_NA = PI / 2
KERNEL_QUADRATIC_COEFFICIENT = 70
UNCONCENTRICITY=0.7
ZOOMED_ANGLE_RANGE = 0.19

MIRROR_LEFT_CENTER = np.array([-1, 0, 0])
MIRROR_RIGHT_CENTER = MIRROR_LEFT_CENTER - np.array([UNCONCENTRICITY, 0, 0])

ALGEBRAIC_EXPRESSIONS_SCALE = 0.8

SCANNING_DOT_TRACKER = ValueTracker(PI - MIRRORS_NA / 2)
SCANNING_DOT_RADIUS_TRACKER = ValueTracker(0.08)
THETA_P_1_TRACKER = ValueTracker(PI / 9)

COLOR_MIRRORS = WHITE
COLOR_INTEGRAL = ORANGE
COLOR_P_1 = RED
DISTANCES_COLOR = BLUE


class Potential(ZoomedScene):
    def construct(self):
        # TODO: choose the real point which is on mirror_right and at angle theta_1
        # Basic system generation:
        mirror_right = Arc(arc_center=MIRROR_RIGHT_CENTER, start_angle=-MIRRORS_NA / 2, angle=MIRRORS_NA, radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        mirror_left = Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=PI - MIRRORS_NA / 2, angle=MIRRORS_NA, radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        p_1_dot = always_redraw(lambda: Dot(color=COLOR_P_1, point=MIRROR_RIGHT_CENTER + MIRRORS_RADIUS * np.cos(THETA_P_1_TRACKER.get_value()) * RIGHT + MIRRORS_RADIUS * np.sin(THETA_P_1_TRACKER.get_value()) * UP))
        p_1_prime_dot = always_redraw(lambda: DashedVMobject(Dot(color=COLOR_P_1, point=MIRROR_LEFT_CENTER + MIRRORS_RADIUS * np.cos(THETA_P_1_TRACKER.get_value() + PI) * RIGHT + MIRRORS_RADIUS * np.sin(THETA_P_1_TRACKER.get_value() + PI) * UP, radius=SCANNING_DOT_RADIUS_TRACKER.get_value(), stroke_width=1, fill_opacity=0)))
        p_0_dot = always_redraw(
            lambda: Dot(
                color=RED,
                radius=SCANNING_DOT_RADIUS_TRACKER.get_value(),
                point=MIRROR_LEFT_CENTER + MIRRORS_RADIUS * np.cos(SCANNING_DOT_TRACKER.get_value()) * RIGHT
                      + MIRRORS_RADIUS * np.sin(SCANNING_DOT_TRACKER.get_value()) * UP,
            )
        )
        p_0_to_p_1_line = always_redraw(lambda: Line(p_0_dot.get_center(), p_1_dot.get_center(), color=WHITE))
        line_length_label = always_redraw(lambda: Tex(f"$r_{{01}}={np.linalg.norm(p_0_dot.get_center() - p_1_dot.get_center()):.2f}$").next_to(p_0_to_p_1_line.get_center(), UP, buff=0.1).rotate(np.arctan2(*(p_1_dot.get_center() - p_0_dot.get_center())[[1, 0]])))
        p_1_label = always_redraw(lambda: Tex(r"$p_{1}$").next_to(p_1_dot.get_center(), UR, buff=0.1))
        p_1_prime_label = always_redraw(lambda: Tex(r"$p^{\prime}_{1}$").next_to(p_1_prime_dot.get_center(), DL, buff=0.0).scale(0.3))
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
                    -np.sin(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value(), THETA_P_1_TRACKER.get_value())),
                ),
                color=COLOR_INTEGRAL,
                stroke_width=1
            )
        )

        integral_representation_path = always_redraw(
            lambda: ParametricFunction(
                lambda t: plane_integral.c2p(*(self.integral_curve(t, THETA_P_1_TRACKER.get_value()))),  # unpack (x,y)
                t_range=(0.0, SCANNING_DOT_TRACKER.get_value()),
                color=COLOR_INTEGRAL,
                stroke_width=1
            ),
        )
        integral_representation = always_redraw(
            lambda: Dot(plane_integral.c2p(*(self.integral_curve(SCANNING_DOT_TRACKER.get_value(), THETA_P_1_TRACKER.get_value()))),
                        color=BLUE,
                        radius=0.03
                        )
        )
        r_01_approximation = Tex(
            r"$r_{01}\approx r_{11^{\prime}}-\frac{r_{11^{\prime}}-R}{2Rr_{11^{\prime}}}\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2}$").to_edge(LEFT).shift(1.5*UP).scale(ALGEBRAIC_EXPRESSIONS_SCALE)#next_to( # This somewhy doesn't work.
            #self.zoomed_display, UP, buff=0.1)

        # p_1 Distances helpers generation:
        p_1_circle = always_redraw(lambda: DashedVMobject(Circle(arc_center=p_1_dot.get_center(), radius=np.linalg.norm(p_1_dot.get_center() - p_1_prime_dot.get_center()), color=DISTANCES_COLOR, stroke_width=0.5), num_dashes=200, dashed_ratio=0.7))
        p_1_circle_radius_line = always_redraw(lambda: DashedLine(p_1_dot.get_center(), p_1_prime_dot.get_center(), color=DISTANCES_COLOR, stroke_width=0.5, dash_length=2 * PI * np.linalg.norm(p_1_dot.get_center() - p_1_prime_dot.get_center()) / 200, dashed_ratio=0.7))
        p_1_circle_radius_label = always_redraw(lambda: Tex(r"$r_{11^{\prime}}$", color=DISTANCES_COLOR).next_to(p_1_circle_radius_line.get_center(), DOWN, buff=0.3).rotate(np.arctan2(*(p_1_dot.get_center() - p_1_prime_dot.get_center())[[1, 0]])))
        distances_group_p_1 = VGroup(p_1_circle, p_1_circle_radius_line, p_1_circle_radius_label, p_1_prime_dot, p_1_prime_label)
        r_11_prime_approximation_label = Tex(r"$r_{11^{\prime}}=2R-u\cos\left(\frac{p_{1}}{R}\right)+\mathcal{O}\left(\left(\frac{u}{R}\right)^{2}\right)$").to_edge(DOWN).scale(ALGEBRAIC_EXPRESSIONS_SCALE)

        # mirror_left Distances helpers generation:
        relevant_radius = MIRRORS_RADIUS - UNCONCENTRICITY
        mirror_left_circle_right_arc = DashedVMobject(Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=-PI/2, angle=PI, radius=relevant_radius, color=DISTANCES_COLOR, stroke_width=0.5), num_dashes=100, dashed_ratio=0.7)
        mirror_left_circle_left_arc = DashedVMobject(Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=PI / 2, angle=PI, radius=relevant_radius, color=DISTANCES_COLOR, stroke_width=0.5), num_dashes=100, dashed_ratio=0.7)
        mirror_left_radius_line = DashedLine(MIRROR_LEFT_CENTER, MIRROR_LEFT_CENTER + (MIRRORS_RADIUS - UNCONCENTRICITY) * RIGHT, color=DISTANCES_COLOR, stroke_width=0.5, dash_length=2 * PI * relevant_radius / 200, dashed_ratio=0.7)
        distances_group_mirror_left = VGroup(mirror_left_circle_right_arc, mirror_left_circle_left_arc, mirror_left_radius_line)  # mirror_left_radius_label

        # Integral result label generation:
        integral_expression = MathTex(r"U\left(\boldsymbol{p}_{1}\right)=\frac{ke^{ikr_{11^{\prime}}}}{4\pi ir_{11^{\prime}}}\intop_{S}U\left(\boldsymbol{p}_{0}\right)e^{-ik\frac{r_{11^{\prime}}-R}{2R\cdot r_{11^{\prime}}}\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2}}dS").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_expression.next_to(box_integral, UP, buff=0.1).to_edge(RIGHT)
        integral_expression_as_convolution = MathTex(r"U\left(\boldsymbol{p}_{1}\right) {{=}} \frac{ke^{ikr_{11^{\prime}}}}{4\pi ir_{11^{\prime}}}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-ik\frac{r_{11^{\prime}}-R}{2R\cdot r_{11^{\prime}}}\boldsymbol{p}_{0}^{2}}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_expression_as_convolution.next_to(box_integral, UP, buff=0.1).to_edge(RIGHT)
        integral_expression_substitute_r_11_prime = MathTex(r" {{=}} \frac{ke^{ik\left(2R-u\cos\left(\frac{s_{1}}{R}\right)\right)}}{8\pi iR}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-ik\frac{R}{4R\cdot}p_{0}^{2}}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_expression_with_separated_potential = MathTex(r"U\left(\boldsymbol{p}_{1}\right) {{=}} \underset{\text{Constant phase}}{\underbrace{\frac{ke^{2iR}}{8\pi iR}}}\cdot\underset{\text{Position dependent phase}}{\underbrace{ {{ e^{-iku\cos\left(\frac{s_{1}}{R}\right)} }} }}\cdot\underset{\text{Convolution}}{\underbrace{ {{ \left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-\frac{ik}{4R}p_{0}^{2}}\right] }} }} }} ").scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_arrow_indicator = Arrow(integral_expression.get_bottom(), plane_integral.c2p(*(self.integral_curve(THETA_P_1_TRACKER.get_value() + PI + ZOOMED_ANGLE_RANGE, THETA_P_1_TRACKER.get_value()))), color=COLOR_INTEGRAL)
        integral_result_group = VGroup(integral_expression, integral_arrow_indicator)

        # Huygens integral introduction
        self.play(Create(mirror_right), Create(mirror_left))
        self.play(Create(p_1_dot), Create(p_0_to_p_1_line), Create(p_0_dot), FadeIn(line_length_label), FadeIn(p_1_label), FadeIn(p_0_label))
        self.add(mirror_right, mirror_left, p_1_dot, p_0_dot, p_0_to_p_1_line, box_integrand, box_integral, phase_representation, integral_representation_path, integral_representation, line_length_label)
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=8, rate_func=linear)

        # Zoomed display and move it to the right place generation:
        self.activate_zooming()
        zoomed_display = self.zoomed_display
        zoomed_display.move_to(p_1_prime_dot.get_center())
        frame = self.zoomed_camera.frame
        frame.move_to(p_1_prime_dot.get_center())
        frame.set_width(1.0)

        # Repeat with zoom animation:
        self.play(Create(distances_group_p_1), mirror_right.animate.set_stroke(width=0.5), mirror_left.animate.set_stroke(width=0.5), p_1_dot.animate.scale(0.5), SCANNING_DOT_RADIUS_TRACKER.animate.set_value(0.04))
        self.play(SCANNING_DOT_TRACKER.animate.set_value(THETA_P_1_TRACKER.get_value() + PI - ZOOMED_ANGLE_RANGE), run_time=1)
        self.play(Write(r_01_approximation), run_time=1)
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=8, rate_func=linear)

        # Interpret algebraic expressions animation:
        self.play(FadeIn(integral_result_group), run_time=2)
        self.play(FadeOut(integral_expression, shift=UP), FadeOut(r_01_approximation),
                  FadeOut(p_0_to_p_1_line), FadeOut(line_length_label), FadeIn(integral_expression_as_convolution, shift=UP), run_time=2)

        # Change discussion to r_01:
        self.play(THETA_P_1_TRACKER.animate.set_value(PI / 6), rate_func=rate_functions.wiggle, run_time=6)
        self.wait(1)
        self.play(Uncreate(distances_group_p_1))
        self.play(Create(distances_group_mirror_left), run_time=2)
        self.play(zoomed_display.animate.move_to(MIRROR_RIGHT_CENTER + MIRRORS_RADIUS * RIGHT), frame.animate.move_to(MIRROR_RIGHT_CENTER + MIRRORS_RADIUS * RIGHT))
        self.play(FadeIn(r_11_prime_approximation_label), run_time=2)

        # Focus on the explicit expression for the integral equation animation:
        self.play(FadeOut(Group(mirror_right, mirror_left, p_1_dot, p_0_dot, box_integrand, box_integral,
                                phase_representation, integral_representation_path, integral_representation, distances_group_mirror_left,
                                integral_arrow_indicator, p_0_label, p_1_label)))
        self.play(FadeOut(frame), FadeOut(zoomed_display))
        self.play(integral_expression_as_convolution.animate.move_to(ORIGIN).to_edge(LEFT))

        eq1 = integral_expression_as_convolution[1]
        eq2 = integral_expression_substitute_r_11_prime[1]
        eq3 = integral_expression_with_separated_potential[1]  # parts are 3, 5
        integral_expression_with_separated_potential[3].set_color(RED)
        integral_expression_with_separated_potential[5].set_color(GREEN)

        integral_expression_substitute_r_11_prime.shift(eq1.get_center() - eq2.get_center())
        integral_expression_with_separated_potential.shift(eq1.get_center() - eq3.get_center())

        self.play(integral_expression_as_convolution.animate.shift(1.5*UP),
                  FadeIn(integral_expression_substitute_r_11_prime, shift=1.5*UP))
        self.play(integral_expression_as_convolution.animate.shift(1.5*UP),
                  integral_expression_substitute_r_11_prime.animate.shift(1.5*UP),
                  FadeIn(integral_expression_with_separated_potential, shift=1.5*UP), run_time=2)
        self.play(FadeOut(integral_expression_as_convolution, shift=3*UP),
                  FadeOut(integral_expression_substitute_r_11_prime, shift=3*UP),
                  FadeOut(r_11_prime_approximation_label),
                  integral_expression_with_separated_potential.animate.to_edge(UP), run_time=2)
        separating_line = Line(np.array([-7.111, 0, 0]), np.array([7.111, 0, 0]), stroke_width=1).next_to(integral_expression_with_separated_potential, DOWN, buff=0.5)
        self.play(Create(separating_line))
        # Schroedinger equations generation:
        schrodinger_1 = MathTex(r"\psi\left(x,t+dt\right) {{ = }} \psi\left(x,t\right)+\partial_{t}\psi\left(x,t\right)\cdot dt+\mathcal{O}\left(dt^{2}\right)").to_corner(DL).shift(0.75*UP).scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        schrodinger_2 = MathTex(r"{{ = }} \left(\mathds{1}-\frac{i\cdot dt}{\hbar}\left(-\hbar^{2}\frac{\nabla^{2}}{2m}+V\left(x\right)\right)\right)\psi\left(x,t\right)+\mathcal{O}\left(dt^{2}\right)", tex_template=tex_template).scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        schrodinger_3 = MathTex(r"{{ = }} \left(\mathds{1}-\frac{i\cdot dt}{\hbar}V\left(x\right)\right)\left(\mathds{1}+idt\frac{\hbar}{2m}\nabla^{2}\right)\psi\left(x,t\right)+\mathcal{O}\left(dt^{2}\right)", tex_template=tex_template).scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        schrodinger_4 = MathTex(r"\psi\left(x,t+dt\right) {{ \approx }}\underset{\text{Position dependent phase}}{\underbrace{ {{ e^{-\frac{i\cdot dt}{\hbar}V\left(x\right)} }} }}\cdot\underset{\text{Convolution}}{\underbrace{ {{ \left[\psi\left(x,t\right)\circledast e^{-i\frac{mx^{2}}{2\hbar dt}}\right] }} }}+\mathcal{O}\left(dt^{2}\right)").scale(ALGEBRAIC_EXPRESSIONS_SCALE)

        schrodinger_2.shift(schrodinger_1[1].get_center() - schrodinger_2[0].get_center())
        schrodinger_3.shift(schrodinger_1[1].get_center() - schrodinger_3[0].get_center())
        schrodinger_4.shift(schrodinger_1[1].get_center() - schrodinger_4[1].get_center())
        schrodinger_4[3].set_color(RED)
        schrodinger_4[5].set_color(GREEN)

        # Schordinger equation analogy animation:
        self.play(FadeIn(schrodinger_1), run_time=1)
        self.play(schrodinger_1.animate.shift(UP), FadeIn(schrodinger_2, shift=UP), run_time=1)
        self.play(schrodinger_1.animate.shift(UP), schrodinger_2.animate.shift(UP), FadeIn(schrodinger_3, shift=UP), run_time=1)
        self.play(schrodinger_1.animate.shift(UP), schrodinger_2.animate.shift(UP), schrodinger_3.animate.shift(UP), FadeIn(schrodinger_4, shift=UP), run_time=1)
        self.play(FadeOut(schrodinger_1, shift=UP), FadeOut(schrodinger_2, shift=UP), FadeOut(schrodinger_3, shift=UP), FadeOut(separating_line, shift=UP),
                  integral_expression_with_separated_potential.animate.move_to(ORIGIN+1*UP),
                  schrodinger_4.animate.move_to(ORIGIN+1*DOWN), run_time=1)
        self.wait(2)
        final_equations = VGroup(integral_expression_with_separated_potential, schrodinger_4)
        self.play(final_equations.animate.scale(0.7).to_corner(UL), run_time=1)
        separating_line.next_to(final_equations, DOWN, buff=0.2)
        self.play(Create(separating_line))

        # Interpret potential animation
        simplified_cavity = VGroup(mirror_left, mirror_right, mirror_left_circle_right_arc, mirror_left_circle_left_arc)
        simplified_cavity.shift(2*DOWN).scale(0.8)
        self.play(FadeIn(simplified_cavity))
        self.play(Wiggle(VGroup(mirror_left_circle_right_arc, mirror_left_circle_left_arc)))

        mirrors_length = MIRRORS_RADIUS * MIRRORS_NA
        flattened_mirror = Line(3*DOWN + LEFT * mirrors_length / 2, 3*DOWN + RIGHT * mirrors_length / 2, color=COLOR_MIRRORS, stroke_width=2)
        flattened_potential = DashedVMobject(ParametricFunction(lambda t: 3*DOWN + t * RIGHT + 1/10 * t**2 * UP,
                                                 t_range=(-mirrors_length / 2, mirrors_length / 2), color=DISTANCES_COLOR, stroke_width=0.5), num_dashes=100, dashed_ratio=0.7,
                                                 )
        self.play(Transform(mirror_right, flattened_mirror),
                  Transform(mirror_left_circle_right_arc, flattened_potential),
                  FadeOut(mirror_left), FadeOut(mirror_left_circle_left_arc))
        self.wait(1)


    @staticmethod
    def integrand_phase_representation(theta, theta_p_1) -> float:
        return KERNEL_QUADRATIC_COEFFICIENT * (theta - (theta_p_1 + PI)) ** 2 - UNCONCENTRICITY * np.cos(theta_p_1) * KERNEL_QUADRATIC_COEFFICIENT

    @staticmethod
    def integral_curve(theta, theta_p_1) -> np.ndarray:
        return 6 * frensel_ax2_integral(
            a=KERNEL_QUADRATIC_COEFFICIENT, x_0=-(theta_p_1 + MIRRORS_NA / 2), x_1=theta - (theta_p_1 + PI), theta_p_1=theta_p_1
        )

    @staticmethod
    def SmallAxesBox(
        side_length=3.0,
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
def frensel_ax2_integral(a, x_0, x_1, theta_p_1):
    factor = np.sqrt(np.pi / (2 * a))
    u0 = np.sqrt(2 * a / np.pi) * x_0
    u1 = np.sqrt(2 * a / np.pi) * x_1
    v0 = fresnel(u0)
    v1 = fresnel(u1)
    v0_complex = v0[0] + 1j * v0[1]
    v1_complex = v1[0] + 1j * v1[1]
    integral_result_without_global_phase = factor * (v1_complex - v0_complex)
    integral_result_complex = integral_result_without_global_phase * np.exp(-1j * UNCONCENTRICITY * np.cos(theta_p_1) * KERNEL_QUADRATIC_COEFFICIENT)
    integral_result_array = np.array([np.real(integral_result_complex), np.imag(integral_result_complex)])
    return integral_result_array


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

# %% # Playground:
if __name__ == "__main__":
    from manim import *

    tex_template = TexTemplate()
    tex_template.add_to_preamble(r"\usepackage{dsfont}")

    ALGEBRAIC_EXPRESSIONS_SCALE = 0.7
    integral_expression = MathTex(
        r"U\left(\boldsymbol{p}_{1}\right)=\frac{ke^{ikr_{11^{\prime}}}}{4\pi ir_{11^{\prime}}}\intop_{S}U\left(\boldsymbol{p}_{0}\right)e^{-ik\frac{r_{11^{\prime}}-R}{2R\cdot r_{11^{\prime}}}\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2}}dS").scale(
        ALGEBRAIC_EXPRESSIONS_SCALE)
    integral_expression_as_convolution = MathTex(
        r"U\left(\boldsymbol{p}_{1}\right) {{=}} \frac{ke^{ikr_{11^{\prime}}}}{4\pi ir_{11^{\prime}}}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-ik\frac{r_{11^{\prime}}-R}{2R\cdot r_{11^{\prime}}}\boldsymbol{p}_{0}^{2}}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)").scale(
        ALGEBRAIC_EXPRESSIONS_SCALE)
    integral_expression_substitute_r_11_prime = MathTex(
        r" {{=}} \frac{ke^{ik\left(2R-u\cos\left(\frac{s_{1}}{R}\right)\right)}}{8\pi iR}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-ik\frac{R}{4R\cdot}p_{0}^{2}}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)").scale(
        ALGEBRAIC_EXPRESSIONS_SCALE)
    integral_expression_with_separated_potential = MathTex(
        r"U\left(\boldsymbol{p}_{1}\right) {{=}}  {{ \underset{\text{Constant phase}}{\underbrace{\frac{ke^{2iR}}{8\pi iR}}} }} \cdot {{ \underset{\text{Position dependent phase}}{\underbrace{e^{-iku\cos\left(\frac{s_{1}}{R}\right)}}}\cdot\underset{\text{Convolution}}{\underbrace{\left[U\left(\boldsymbol{p}_{0}\right)\circledast e^{-\frac{ik}{4R}p_{0}^{2}}\right]}} }} ").scale(
        ALGEBRAIC_EXPRESSIONS_SCALE)

    # eq3 = integral_expression_with_separated_potential[0]

