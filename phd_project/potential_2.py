from manim import *

class Potential(ZoomedScene):
    def construct(self) -> None:
        # self.introduction_TOC()
        self.resonators_overview()

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
        wiggle_tracker = ValueTracker(0)
        MIRRORS_RADIUS = 1.5
        MIRRORS_NA = PI / 2
        UNCONCENTRICITY = 0.35
        COLOR_MIRRORS = WHITE

        MIRROR_LEFT_CENTER = np.array([-3, 0, 0])
        MIRROR_RIGHT_CENTER = MIRROR_LEFT_CENTER - np.array([UNCONCENTRICITY, 0, 0])

        MODE_NA = 0.35   # numerical aperture of the mode — increase to focus tighter
        MODE_W0 = 0.18   # beam waist radius (scene units)

        title = Text("Resonators").scale(0.8).to_edge(UP)
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
        quantum_system_gaussian = always_redraw(lambda: hypothetical_quantum_system.plot(
            lambda x: np.exp(-x**2) + 0.05 * np.sin(0.4 * x - wiggle_tracker.get_value()),
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
            lambda x, s=spot_size_left: np.exp(-x ** 2 / (2 * s ** 2)) + 0.05 * np.sin(0.4 * x - wiggle_tracker.get_value()),
            color=RED,
        ))
        self.add(title, mirror_left, mirror_right, mode, vertical_line_separator,
                 hypothetical_quantum_system, quantum_system_gaussian,
                 mirror_field_group, mirror_field_gaussian)
        self.play(wiggle_tracker.animate.set_value(3), run_time=3)




