from manim import *

class Potential(ZoomedScene):
    def construct(self) -> None:
        self.introduction_TOC()
        self.resonators_overview()
        self.potential_overview()

    def introduction_TOC(self):
        title = Text("Wave equation solutions in non-paraxial resonators").scale(0.8).to_edge(UP)
        sub_title = Text("What are we going to have?").next_to(title, DOWN, buff=0.5).scale(0.7)
        toc = VGroup(
            VGroup(Dot(stroke_color=BLUE), Text("Classical physics").scale(0.7)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Text("Not a quantum mechanical problem").scale(0.7)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Text("Geometry").scale(0.7)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(), Text("Swag").scale(0.7)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(sub_title, DOWN, buff=0.5).to_edge(LEFT).shift(0.5*RIGHT)

        self.play(FadeIn(title))
        self.play(FadeIn(sub_title))
        self.play(FadeIn(toc[0]))
        self.play(FadeIn(toc[1]))
        self.play(FadeIn(toc[2]))
        self.play(FadeIn(toc[3]))
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
        diagonal_stroke_over_paraxial = Line(start=paraxial_approximation_equation.get_corner(DOWN + RIGHT), end=paraxial_approximation_equation.get_corner(UP + LEFT), color=RED)
        arrow = Arrow(start=helmholtz_equation.get_top(), end=mode.get_center() - 0.3*UP, buff=0.1)
        self.play(FadeIn(title))
        self.play(Create(mirror_left), Create(mirror_right))
        self.play(Create(mode))
        self.play(Create(arrow))
        self.play(FadeIn(helmholtz_equation))
        self.play(Create(paraxial_approximation_equation))
        self.play(MODE_NA.animate.set_value(0.3), run_time=3)
        self.play(Create(diagonal_stroke_over_paraxial))
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
        self.play(FadeIn(title))
        self.play(FadeIn(subtitle))
        self.play(Create(mirror_left), Create(mirror_right))
        self.play(Create(mode))
        label_mirror = MathTex(r"E(y)").scale(0.8).next_to(mirror_field_group, UP, buff=0.2)
        label_quantum = MathTex(r"\psi(x)").scale(0.8).next_to(hypothetical_quantum_system, UP, buff=0.2)

        self.play(Create(mirror_field_group), Create(mirror_field_gaussian))
        self.play(FadeIn(label_mirror))
        self.play(wiggle_tracker.animate.set_value(30), run_time=5)
        wiggle_tracker.set_value(0)
        self.play(Create(vertical_line_separator))
        self.play(Create(hypothetical_quantum_system), Create(quantum_system_gaussian))
        self.play(FadeIn(label_quantum))
        self.play(wiggle_tracker.animate.set_value(30), run_time=5)
        # Fade out everything:
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(mirror_left), FadeOut(mirror_right), FadeOut(mode), FadeOut(vertical_line_separator),
                  FadeOut(hypothetical_quantum_system), FadeOut(quantum_system_gaussian), FadeOut(mirror_field_group), FadeOut(mirror_field_gaussian),
                  FadeOut(label_mirror), FadeOut(label_quantum))






