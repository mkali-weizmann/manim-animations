from manim import *
from scipy.special import fresnel
from manim_slides import Slide

tex_template = TexTemplate()
tex_template.add_to_preamble(r"\usepackage{dsfont}")

UNCONCENTRICITY = 0.7
KERNEL_QUADRATIC_COEFFICIENT = 120
MIRRORS_NA = PI / 2

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

class Potential(ZoomedScene, Slide):
    def construct(self) -> None:
        self.wait(0.1)
        self.next_slide()

        self.introduction_TOC()

        self.resonators_overview()

        self.potential_overview()

        self.lemma_intro()

        self.derivation()

        self.conclusions()

    def introduction_TOC(self):
        title = Text("Wave equation solutions in non-paraxial resonators").scale(0.8).to_edge(UP)
        sub_title = Text("What are we going to have?").next_to(title, DOWN, buff=0.5).scale(0.7)
        toc = VGroup(
            VGroup(Dot(stroke_color=BLUE), Text("Classical physics").scale(0.7)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Text("Not a quantum mechanical problem").scale(0.7)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Text("Geometry").scale(0.7)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Text("Swag").scale(0.7)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(sub_title, DOWN, buff=0.5).to_edge(LEFT).shift(0.5*RIGHT)

        self.play(FadeIn(title, shift=UP))
        self.next_slide()
        self.play(FadeIn(sub_title, shift=UP))
        self.next_slide()
        self.play(FadeIn(toc[0], shift=UP))
        self.next_slide()
        self.play(FadeIn(toc[1], shift=UP))
        self.next_slide()
        self.play(FadeIn(toc[2], shift=UP))
        self.next_slide()
        self.play(FadeIn(toc[3], shift=UP))
        self.next_slide()
        # Fade out everything:
        self.play(FadeOut(title), FadeOut(sub_title), FadeOut(toc))

    def resonators_overview(self):
        title = Text("Resonators").scale(0.8).to_edge(UP)
        MIRRORS_RADIUS = 3
        MIRRORS_NA = PI / 2
        UNCONCENTRICITY = 0.35
        COLOR_MIRRORS = WHITE
        VERTICAL_SHIFT = -0.5

        MIRROR_LEFT_CENTER = np.array([UNCONCENTRICITY/2, VERTICAL_SHIFT, 0])
        MIRROR_RIGHT_CENTER = MIRROR_LEFT_CENTER - np.array([UNCONCENTRICITY, 0, 0])

        MODE_NA = ValueTracker(0.05)

        mirror_right = Arc(arc_center=MIRROR_RIGHT_CENTER, start_angle=-MIRRORS_NA / 2, angle=MIRRORS_NA,
                           radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        mirror_left = Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=PI - MIRRORS_NA / 2, angle=MIRRORS_NA,
                          radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)

        dx = MIRRORS_RADIUS / 15
        x_left  = MIRROR_LEFT_CENTER[0]  - MIRRORS_RADIUS + dx   # left  mirror vertex
        x_right = MIRROR_RIGHT_CENTER[0] + MIRRORS_RADIUS - dx   # right mirror vertex
        x_waist = (x_left + x_right) / 2                    # beam waist at cavity midpoint

        def make_mode_curve(sign):
            na = MODE_NA.get_value()
            w0 = 1 / na / 100
            z_r = w0 / np.tan(np.arcsin(na))
            return ParametricFunction(
                lambda t, _w0=w0, _zr=z_r: np.array(
                    [t, sign * _w0 * np.sqrt(1 + ((t - x_waist) / _zr) ** 2) + VERTICAL_SHIFT, 0]
                ),
                t_range=(x_left, x_right),
                color=YELLOW,
            )

        mode_upper = always_redraw(lambda: make_mode_curve(+1))
        mode_lower = always_redraw(lambda: make_mode_curve(-1))
        mode = VGroup(mode_upper, mode_lower)

        helmholtz_equation = MathTex(r"\nabla^2 E + k^2 E = 0").to_edge(DOWN).shift(0.5*RIGHT)
        paraxial_approximation_equation = MathTex(r"\sin \theta \approx \theta \approx \tan \theta").next_to(helmholtz_equation, RIGHT, buff=0.5)
        diagonal_stroke_over_paraxial = Line(start=paraxial_approximation_equation.get_corner(DOWN + LEFT), end=paraxial_approximation_equation.get_corner(UP + RIGHT), color=RED)
        arrow = Arrow(start=helmholtz_equation.get_top(), end=mode.get_center() - 0.3*UP, buff=0.1)
        self.play(FadeIn(title, shift=UP))
        self.play(Create(mirror_left), Create(mirror_right))
        self.play(Create(mode))
        self.next_slide()
        self.play(Create(arrow))
        self.play(FadeIn(helmholtz_equation, shift=UP))
        self.next_slide()
        self.play(Create(paraxial_approximation_equation))
        self.next_slide()
        self.play(MODE_NA.animate.set_value(0.3), run_time=3)
        self.play(Create(diagonal_stroke_over_paraxial))
        self.next_slide()
        # Fade out everything:
        self.play(FadeOut(title), FadeOut(mirror_left), FadeOut(mirror_right), FadeOut(mode), FadeOut(helmholtz_equation), FadeOut(arrow), FadeOut(paraxial_approximation_equation), FadeOut(diagonal_stroke_over_paraxial))

    def potential_overview(self):
        wiggle_tracker = ValueTracker(0)
        MIRRORS_RADIUS = 1.5
        MIRRORS_NA = PI / 2
        UNCONCENTRICITY = 0.35
        COLOR_MIRRORS = WHITE

        MIRROR_LEFT_CENTER = np.array([-3, 0, 0])
        MIRROR_RIGHT_CENTER = MIRROR_LEFT_CENTER - np.array([UNCONCENTRICITY, 0, 0])

        MODE_NA = 0.35   # numerical aperture of the mode — increase to focus tighter
        MODE_W0 = 0.18   # beam waist radius (scene units)

        title = Text("Our Claim").scale(0.8).to_edge(UP)
        subtitle = Text("The eigenmodes on the end mirrors satisfy a schrodinger equation").scale(0.7).next_to(title, DOWN, buff=0.5)
        mirror_right = Arc(arc_center=MIRROR_RIGHT_CENTER, start_angle=-MIRRORS_NA / 2, angle=MIRRORS_NA,
                           radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        mirror_left = Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=PI - MIRRORS_NA / 2, angle=MIRRORS_NA,
                          radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)

        # Gaussian mode as two hyperbolas: w(z) = w0 * sqrt(1 + (z/z_R)^2)
        # z_R (Rayleigh range) is derived from NA = sin(divergence half-angle), tan(theta) = w0/z_R
        dx = MIRRORS_RADIUS / 15
        x_left  = MIRROR_LEFT_CENTER[0]  - MIRRORS_RADIUS + dx   # left  mirror vertex

        x_right = MIRROR_RIGHT_CENTER[0] + MIRRORS_RADIUS - dx   # right mirror vertex
        x_waist = (x_left + x_right) / 2                    # beam waist at cavity midpoint
        z_R = MODE_W0 / np.tan(np.arcsin(MODE_NA))          # Rayleigh range

        mode_upper = ParametricFunction(
            lambda t: np.array([t, MODE_W0 * np.sqrt(1 + ((t - x_waist) / z_R) ** 2), 0]),
            t_range=(x_left, x_right),
            color=YELLOW,
        )
        mode_lower = ParametricFunction(
            lambda t: np.array([t, -MODE_W0 * np.sqrt(1 + ((t - x_waist) / z_R) ** 2), 0]),
            t_range=(x_left, x_right),
            color=YELLOW,
        )
        mode = VGroup(mode_upper, mode_lower)

        vertical_line_separator = Line(start=UP, end=DOWN)

        hypothetical_quantum_system = Axes(x_range=[-3, 3], y_range=[0, 1.5], x_length=4, y_length=1.5,
                                           axis_config={"include_tip": False}).shift(3*RIGHT)
        k_wiggle = 6
        A_wiggle = 0.015
        omega_wiggle = 0.2
        wiggle_function = lambda x: A_wiggle * np.sin(k_wiggle * x - omega_wiggle * wiggle_tracker.get_value()) * (1 / (1 + np.exp(-wiggle_tracker.get_value()+6)) - 1 / (1 + np.exp(-wiggle_tracker.get_value()+24)))
        quantum_system_gaussian = always_redraw(lambda: hypothetical_quantum_system.plot(
            lambda x: np.exp(-x**2) + wiggle_function(x),
            color=RED,
        ))

        # Field intensity profile on the left mirror: I(y) = exp(-y²/2w²), rotated so domain is vertical
        spot_size_left = 0.8
        mirror_field_axes = Axes(
            x_range=[-3, 3],
            y_range=[0, 1.2],
            x_length=2.4,
            y_length=1.1,
            axis_config={"include_tip": False},
        )
        mirror_field_axes.y_axis.set_stroke(opacity=0)
        mirror_field_group = VGroup(mirror_field_axes)
        mirror_field_group.rotate(-PI / 2)
        mirror_field_group.next_to(mirror_left, LEFT, buff=0.3)
        mirror_field_group.set_y(0)

        mirror_field_gaussian = always_redraw(lambda: mirror_field_axes.plot(
            lambda x, s=spot_size_left: np.exp(-x ** 2 / (2 * s ** 2)) + wiggle_function(x),
            color=RED,
        ))
        # self.add(title, mirror_left, mirror_right, mode, vertical_line_separator,
        #          hypothetical_quantum_system, quantum_system_gaussian,
        #          mirror_field_group, mirror_field_gaussian)
        self.play(FadeIn(title, shift=UP))
        self.next_slide()
        self.play(FadeIn(subtitle, shift=UP))
        self.next_slide()
        self.play(Create(mirror_left), Create(mirror_right))
        self.play(Create(mode))
        label_mirror = MathTex(r"E(y)").scale(0.8).next_to(mirror_field_group, UP, buff=0.2)
        label_quantum = MathTex(r"\psi(x)").scale(0.8).next_to(hypothetical_quantum_system, UP, buff=0.2)

        self.next_slide()
        self.play(Create(mirror_field_group), Create(mirror_field_gaussian))
        self.play(FadeIn(label_mirror))
        self.play(wiggle_tracker.animate.set_value(30), run_time=5)
        wiggle_tracker.set_value(0)
        self.next_slide()
        self.play(Create(vertical_line_separator))
        self.play(Create(hypothetical_quantum_system), Create(quantum_system_gaussian))
        self.play(FadeIn(label_quantum))
        self.play(wiggle_tracker.animate.set_value(30), run_time=5)
        # Fade out everything:
        self.next_slide()
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(mirror_left), FadeOut(mirror_right), FadeOut(mode), FadeOut(vertical_line_separator),
                  FadeOut(hypothetical_quantum_system), FadeOut(quantum_system_gaussian), FadeOut(mirror_field_group), FadeOut(mirror_field_gaussian),
                  FadeOut(label_mirror), FadeOut(label_quantum))

    def lemma_intro(self):
        title = Text("Assumption").scale(0.8).to_edge(UP)
        statement = Text("A field that returns to itself after one roundtrip\nin the resonator will be an eigenmode").scale(0.7).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(title, shift=UP))
        self.next_slide()
        self.play(FadeIn(statement, shift=UP))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(statement))

    def derivation(self):
        ALGEBRAIC_EXPRESSIONS_SCALE = 0.8
        MIRRORS_RADIUS = 3
        MIRRORS_NA = PI / 2
        ZOOMED_ANGLE_RANGE = 0.19
        MIRROR_VERTICAL_SHIFT = -1.2
        MIRROR_LEFT_CENTER = np.array([-1, MIRROR_VERTICAL_SHIFT, 0])
        MIRROR_RIGHT_CENTER = MIRROR_LEFT_CENTER - np.array([UNCONCENTRICITY, 0, 0])

        ALGEBRAIC_EXPRESSIONS_SCALE = 0.8

        SCANNING_DOT_TRACKER = ValueTracker(PI - MIRRORS_NA / 2)
        SCANNING_DOT_RADIUS_TRACKER = ValueTracker(0.08)
        THETA_P_1_TRACKER = ValueTracker(PI / 9)

        COLOR_MIRRORS = WHITE
        COLOR_INTEGRAL = ORANGE
        COLOR_P_1 = RED
        COLOR_POTENTIAL = RED
        COLOR_KINETIC_TERM = GREEN

        title = Text("Derivation").scale(0.8).to_edge(UP)
        subtitle = Text("We use the Huygens integral:").scale(0.7).next_to(title, DOWN, buff=0.5)
        huygens_integral_equation = MathTex(r"U\left(\boldsymbol{r}_{1}\right)=\frac{1}{i\lambda}\iint_{S}U\left(\boldsymbol{r}_{0}\right)\frac{e^{ikr_{01}}}{r_{01}}\cos\theta d\boldsymbol{r}_{0}").scale(0.7).next_to(subtitle, DOWN, buff=0.5)

        self.play(FadeIn(title, shift=UP))
        self.next_slide()
        self.play(FadeIn(subtitle, shift=UP))
        self.play(FadeIn(huygens_integral_equation, shift=UP))
        self.next_slide()
        self.play(FadeOut(subtitle), title.animate.scale(0.8).to_corner(UL), huygens_integral_equation.animate.scale(0.8).to_corner(UL).shift(0.5*DOWN))

        # Basic system generation:
        mirror_right = Arc(arc_center=MIRROR_RIGHT_CENTER, start_angle=-MIRRORS_NA / 2, angle=MIRRORS_NA,
                           radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        mirror_left = Arc(arc_center=MIRROR_LEFT_CENTER, start_angle=PI - MIRRORS_NA / 2, angle=MIRRORS_NA,
                          radius=MIRRORS_RADIUS, color=COLOR_MIRRORS)
        p_1_prime_dot = always_redraw(lambda: DashedVMobject(Dot(color=COLOR_P_1,
                                                                 point=MIRROR_LEFT_CENTER + MIRRORS_RADIUS * np.cos(
                                                                     THETA_P_1_TRACKER.get_value() + PI) * RIGHT + MIRRORS_RADIUS * np.sin(
                                                                     THETA_P_1_TRACKER.get_value() + PI) * UP,
                                                                 radius=SCANNING_DOT_RADIUS_TRACKER.get_value(),
                                                                 stroke_width=1, fill_opacity=0)))
        p_1_dot = always_redraw(
            lambda: Dot(color=COLOR_P_1, point=find_intersection_with_ray(ray_origin=p_1_prime_dot.get_center(),
                                                                          circle_origin=MIRROR_RIGHT_CENTER,
                                                                          angle=THETA_P_1_TRACKER.get_value(),
                                                                          circle_radius=MIRRORS_RADIUS))
            )
        p_0_dot = always_redraw(
            lambda: Dot(
                color=RED,
                radius=SCANNING_DOT_RADIUS_TRACKER.get_value(),
                point=MIRROR_LEFT_CENTER + MIRRORS_RADIUS * np.cos(SCANNING_DOT_TRACKER.get_value()) * RIGHT
                      + MIRRORS_RADIUS * np.sin(SCANNING_DOT_TRACKER.get_value()) * UP,
            )
        )
        p_0_to_p_1_line = always_redraw(lambda: Line(p_0_dot.get_center(), p_1_dot.get_center(), color=WHITE))
        line_length_label = always_redraw(
            lambda: Tex(f"$r_{{01}}={np.linalg.norm(p_0_dot.get_center() - p_1_dot.get_center()):.2f}$").next_to(
                p_0_to_p_1_line.get_center(), UP, buff=0.1).rotate(
                np.arctan2(*(p_1_dot.get_center() - p_0_dot.get_center())[[1, 0]])))
        p_1_label = always_redraw(lambda: Tex(r"$p_{1}$").next_to(p_1_dot.get_center(), UR, buff=0.1))
        p_1_prime_label = Tex(r"$p^{\prime}_{1}$").next_to(p_1_prime_dot.get_center(), DL, buff=0.0).scale(0.3)
        p_0_label = always_redraw(lambda: Tex(r"$p_{0}$").next_to(p_0_dot, LEFT, buff=0.1))

        # Integrand and integral representations' generation:
        box_integrand = self.SmallAxesBox(2.5).to_corner(DR)
        box_integrand_label = always_redraw(
            lambda: Tex(f"$e^{{ik\cdot{np.linalg.norm(p_0_dot.get_center() - p_1_dot.get_center()):.2f}}}$").next_to(  #
                box_integrand, UP, buff=0.1))
        box_integral = self.SmallAxesBox(2.5).next_to(box_integrand, UP, buff=1)
        box_integral_label = Tex(r"$\iint_{S}\ldots d\boldsymbol{r}_{0}$").next_to(box_integral, UP, buff=0.1)
        plane_integrand = box_integrand[0]  # extract the NumberPlane
        plane_integral = box_integral[0]  # extract the NumberPlane
        phase_representation = always_redraw(
            lambda: Line(
                plane_integrand.c2p(0, 0),
                plane_integrand.c2p(
                    np.cos(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value(),
                                                               THETA_P_1_TRACKER.get_value())),
                    -np.sin(self.integrand_phase_representation(SCANNING_DOT_TRACKER.get_value(),
                                                                THETA_P_1_TRACKER.get_value())),
                ),
                color=COLOR_INTEGRAL,
                stroke_width=1
            )
        )
        phase_representation_dot = always_redraw(
            lambda: Dot(plane_integral.c2p(
                *(self.integral_curve(SCANNING_DOT_TRACKER.get_value(), THETA_P_1_TRACKER.get_value()))),
                        color=BLUE,
                        radius=0.03
                        ))

        integral_representation_path = always_redraw(
            lambda: ParametricFunction(
                lambda t: plane_integral.c2p(*(self.integral_curve(t, THETA_P_1_TRACKER.get_value()))),  # unpack (x,y)
                t_range=(0.0, SCANNING_DOT_TRACKER.get_value()),
                color=COLOR_INTEGRAL,
                stroke_width=1
            ),
        )
        integral_representation = always_redraw(
            lambda: Dot(plane_integral.c2p(
                *(self.integral_curve(SCANNING_DOT_TRACKER.get_value(), THETA_P_1_TRACKER.get_value()))),
                        color=BLUE,
                        radius=0.03
                        )
        )
        r_01_approximation = MathTex(
            r"r_{01}\approx {{ V\left(\boldsymbol{p}_{1}^{\prime}\right) }} + {{ H\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2} }}").scale(ALGEBRAIC_EXPRESSIONS_SCALE).next_to(mirror_left, UP)
        r_01_approximation[1].set_color(COLOR_POTENTIAL)
        r_01_approximation[3].set_color(COLOR_KINETIC_TERM)
        planes_group = VGroup(box_integrand, box_integral, plane_integrand, plane_integral, box_integrand_label, box_integral_label, phase_representation, phase_representation_dot, integral_representation_path, integral_representation)


        # p_1 Distances helpers generation:
        p_1_circle = always_redraw(lambda: DashedVMobject(Arc(arc_center=p_1_dot.get_center(), radius=float(np.linalg.norm(
            p_1_dot.get_center() - p_1_prime_dot.get_center())), color=COLOR_KINETIC_TERM, stroke_width=0.5, start_angle=PI / 2, angle=PI),
                                                          num_dashes=200, dashed_ratio=0.7))
        p_1_circle_radius_line = always_redraw(
            lambda: DashedLine(p_1_dot.get_center(), p_1_prime_dot.get_center(), color=COLOR_POTENTIAL,
                               stroke_width=0.5, dash_length=2 * PI * float(np.linalg.norm(
                    p_1_dot.get_center() - p_1_prime_dot.get_center())) / 200, dashed_ratio=0.7))
        p_1_circle_radius_label = always_redraw(
            lambda: Tex(r"$V\left(\boldsymbol{p}_{1}^{\prime}\right)$", color=COLOR_POTENTIAL).next_to(p_1_circle_radius_line.get_center(), UP,
                                                                             buff=0.3).rotate(
                np.arctan2(*(p_1_dot.get_center() - p_1_prime_dot.get_center())[[1, 0]])).set_opacity(0.8).set_z_index(-1))
        distances_group_p_1 = VGroup(p_1_circle, p_1_circle_radius_line, p_1_circle_radius_label, p_1_prime_dot,
                                     p_1_prime_label)

        # Integral result label generation:
        huygens_substituted_expansion = MathTex(
            r"U\left(\boldsymbol{p}_{1}\right)\propto {{ e^{ikV\left(\boldsymbol{p}_{1}^{\prime}\right)} }}\cdot\iint_{S}U\left(\boldsymbol{p}_{0}\right) {{ e^{-ikH\left(\boldsymbol{p}_{0}-\boldsymbol{p}_{1}^{\prime}\right)^{2}} }}dS").scale(
            ALGEBRAIC_EXPRESSIONS_SCALE)
        huygens_substituted_expansion.move_to(huygens_integral_equation.get_center()).align_to(huygens_integral_equation, LEFT)
        huygens_substituted_expansion[1].set_color(COLOR_POTENTIAL)
        huygens_substituted_expansion[3].set_color(COLOR_KINETIC_TERM)
        integral_expression_as_convolution = MathTex(
            r"U\left(\boldsymbol{p}_{1}\right)\propto {{ e^{ikV\left(\boldsymbol{p}_{1}^{\prime}\right)} }}\cdot\left[U\left(\boldsymbol{p}_{0}\right)\circledast {{ e^{-ikH\left(\boldsymbol{p}_{0}\right)^{2}} }}\right]\left(\boldsymbol{p}_{1}^{\prime}\right)").scale(
            ALGEBRAIC_EXPRESSIONS_SCALE)
        integral_expression_as_convolution.move_to(huygens_integral_equation.get_center()).align_to(huygens_integral_equation, LEFT)
        integral_expression_as_convolution[1].set_color(COLOR_POTENTIAL)
        integral_expression_as_convolution[3].set_color(COLOR_KINETIC_TERM)

                # Huygens integral introduction
        self.play(Create(mirror_right), Create(mirror_left))
        self.play(Create(p_1_dot), Create(p_0_to_p_1_line), Create(p_0_dot),
                  FadeIn(p_1_label), FadeIn(p_0_label), Create(planes_group))
        self.next_slide()
        self.play(FadeIn(line_length_label))
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=8, rate_func=linear)

        # Zoomed display and move it to the right place generation:
        self.activate_zooming()
        zoomed_display = self.zoomed_display
        zoomed_display.move_to(p_1_prime_dot.get_center())
        frame = self.zoomed_camera.frame
        frame.move_to(p_1_prime_dot.get_center())
        frame.set_width(1.0)

        # Repeat with zoom animation:
        self.play(Create(distances_group_p_1), mirror_right.animate.set_stroke(width=0.5),
                  mirror_left.animate.set_stroke(width=0.5), p_1_dot.animate.scale(0.5),
                  SCANNING_DOT_RADIUS_TRACKER.animate.set_value(0.04))
        p_1_prime_label.add_updater(lambda m: m.become(Tex(r"$p^{\prime}_{1}$").next_to(p_1_prime_dot.get_center(), DL, buff=0.0).scale(0.3)))
        self.next_slide()
        self.play(SCANNING_DOT_TRACKER.animate.set_value(THETA_P_1_TRACKER.get_value() + PI - ZOOMED_ANGLE_RANGE),
                  run_time=1)
        self.next_slide()
        self.play(Write(r_01_approximation), run_time=1)
        self.next_slide()
        self.play(SCANNING_DOT_TRACKER.animate.set_value(PI + MIRRORS_NA / 2), run_time=8, rate_func=linear)
        self.play(THETA_P_1_TRACKER.animate.set_value(PI / 6), rate_func=rate_functions.wiggle, run_time=6)
        self.next_slide()
        self.play(FadeOut(huygens_integral_equation, shift=UP), FadeIn(huygens_substituted_expansion, shift=UP))
        self.next_slide()
        self.play(FadeOut(huygens_substituted_expansion, shift=UP), FadeIn(integral_expression_as_convolution, shift=UP))
        self.next_slide()
        self.play(FadeOut(distances_group_p_1, mirror_right, mirror_left,
                          p_1_dot, p_0_dot, p_0_to_p_1_line, line_length_label, planes_group, r_01_approximation, p_0_label, p_1_label),
                  FadeOut(frame), FadeOut(zoomed_display))

        separating_line = Line(np.array([-7.111, 0, 0]), np.array([7.111, 0, 0]), stroke_width=1).next_to(
            integral_expression_as_convolution, DOWN, buff=0.5)
        # self.play(Create(separating_line))
        # Schroedinger equations generation:
        schrodinger_1 = MathTex(
            r"\psi\left(x,t+dt\right) {{ = }} \psi\left(x,t\right)+\partial_{t}\psi\left(x,t\right)\cdot dt+\mathcal{O}\left(dt^{2}\right)").to_corner(
            DL).shift(0.75 * UP).scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        schrodinger_2 = MathTex(
            r"{{ = }} \left(\mathds{1}-\frac{i\cdot dt}{\hbar}\left( {{ V\left(x\right) }} - {{ \hbar^{2}\frac{\nabla^{2}}{2m} }}  \right)\right)\psi\left(x,t\right)+\mathcal{O}\left(dt^{2}\right)",
            tex_template=tex_template).scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        schrodinger_3 = MathTex(
            r"{{ = }} \left( {{ \mathds{1}-\frac{i\cdot dt}{\hbar}V\left(x\right) }} \right)\left( {{ \mathds{1}+idt\frac{\hbar}{2m}\nabla^{2} }} \right)\psi\left(x,t\right)+\mathcal{O}\left(dt^{2}\right)",
            tex_template=tex_template).scale(ALGEBRAIC_EXPRESSIONS_SCALE)
        schrodinger_4 = MathTex(
            r"\psi\left(x,t+dt\right) {{ \approx }}\underset{\text{Position dependent phase}}{\underbrace{ {{ e^{-\frac{i\cdot dt}{\hbar}V\left(x\right)} }} }}\cdot\underset{\text{Convolution}}{\underbrace{\left[\psi\left(x,t\right)\circledast {{ e^{-i\frac{mx^{2}}{2\hbar dt}} }}\right]}}+\mathcal{O}\left(dt^{2}\right)").scale(
            ALGEBRAIC_EXPRESSIONS_SCALE)

        schrodinger_2.shift(schrodinger_1[1].get_center() - schrodinger_2[0].get_center())
        schrodinger_3.shift(schrodinger_1[1].get_center() - schrodinger_3[0].get_center())
        schrodinger_4.shift(schrodinger_1[1].get_center() - schrodinger_4[1].get_center())
        schrodinger_2[2].set_color(COLOR_POTENTIAL)
        schrodinger_2[4].set_color(COLOR_KINETIC_TERM)
        schrodinger_3[2].set_color(COLOR_POTENTIAL)
        schrodinger_3[4].set_color(COLOR_KINETIC_TERM)
        schrodinger_4[3].set_color(COLOR_POTENTIAL)
        schrodinger_4[5].set_color(COLOR_KINETIC_TERM)

        # Schordinger equation analogy animation:
        self.play(FadeIn(schrodinger_1, shift=UP), run_time=1)
        self.next_slide()
        self.play(schrodinger_1.animate.shift(UP), FadeIn(schrodinger_2, shift=UP), run_time=1)
        self.next_slide()
        self.play(schrodinger_1.animate.shift(UP), schrodinger_2.animate.shift(UP), FadeIn(schrodinger_3, shift=UP),
                  run_time=1)
        self.next_slide()
        self.play(schrodinger_1.animate.shift(UP), schrodinger_2.animate.shift(UP), schrodinger_3.animate.shift(UP),
                  FadeIn(schrodinger_4, shift=UP), run_time=1)
        self.next_slide()
        self.play(FadeOut(schrodinger_1, shift=UP), FadeOut(schrodinger_2, shift=UP), FadeOut(schrodinger_3, shift=UP),
                  FadeOut(separating_line, shift=UP),
                  integral_expression_as_convolution.animate.move_to(ORIGIN + 1 * UP),
                  schrodinger_4.animate.move_to(ORIGIN + 1 * DOWN), run_time=1)
        self.next_slide()
        self.play(FadeOut(integral_expression_as_convolution), FadeOut(schrodinger_4), FadeOut(title))

    def conclusions(self):
        title = Text("Conclusions").scale(0.8).to_edge(UP)

        conclusions_bullets = VGroup(
            VGroup(Dot(stroke_color=BLUE), Tex("Aberrations $\iff$ Non-harmonic potential").scale(0.9)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Tex("Unstable resonators $\iff$ Non-confining potential").scale(0.9)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Tex("Thermal lensing $\iff$ Potential changes and distorts in time" ).scale(0.9)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Tex("Optical elements imperfections $\iff$ Also potential distortion").scale(0.9)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).next_to(title, DOWN, buff=0.5).to_edge(LEFT).shift(0.5 * RIGHT)

        self.play(FadeIn(title, shift=UP))
        self.next_slide()
        self.play(FadeIn(conclusions_bullets[0]))
        self.next_slide()
        self.play(FadeIn(conclusions_bullets[1]))
        self.next_slide()
        self.play(FadeIn(conclusions_bullets[2]))
        self.next_slide()
        self.play(FadeIn(conclusions_bullets[3]))
        self.next_slide()
        # Fade out everything:
        self.play(FadeOut(title), FadeOut(conclusions_bullets))




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







