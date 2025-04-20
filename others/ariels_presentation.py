from manim import *
import numpy as np
from scipy import special
from manim_slides import Slide


config.background_color = WHITE

# Constants
MHz = 1.0
Gamma = 6 * MHz
OD = 100

# Function to calculate transmission
def calculate_transmission(Delta_p, Gamma, Omega_776, Delta_776, gamma_gd,
                           Omega_480, Delta_480, gamma_gs,
                           Omega_1260, Delta_1260, gamma_gr, OD):
    delta_d = Delta_p + Delta_776
    delta_s = Delta_p + Delta_480
    delta_r = Delta_p + Delta_776 + Delta_1260

    sus = 1j / (
            Gamma - 1j * Delta_p
            + (Omega_480 ** 2) / (gamma_gs - 1j * delta_s)
            + (Omega_776 ** 2) / (gamma_gd - 1j * delta_d + (Omega_1260 ** 2) / (gamma_gr - 1j * delta_r))
    )
    Trans = np.exp(-OD * np.imag(sus))
    return Trans


class CombinedScene(Slide):
    def construct(self):
        distance_factor = 6
        # Energy levels (move to the left edge)
        level_5S = Line(LEFT, RIGHT, color=BLACK).scale(0.7).shift(DOWN * 3 + LEFT * 3.5)
        level_5P = Line(LEFT, RIGHT, color=BLACK).scale(0.7).next_to(level_5S, distance_factor * (UP + 0.01 * RIGHT)).shift(1*LEFT)
        level_5D = Line(LEFT, RIGHT, color=BLACK).scale(0.7).next_to(level_5P, distance_factor * UP)
        level_100S = Line(LEFT, RIGHT, color=BLACK).scale(0.7).next_to(level_5D, distance_factor * (UP + 0.15 * LEFT))
        level_99P = Line(LEFT, RIGHT, color=BLACK).scale(0.7).next_to(level_5D, distance_factor * (UP + 0.05 * RIGHT)).shift(LEFT * 0.3)

        label_minus = Tex(r"$\left|-\right\rangle$", color=BLACK).next_to(level_5P, RIGHT).shift(UP * 0.3)
        label_plus = Tex(r"$\left|+\right\rangle$", color=BLACK).next_to(level_5D, RIGHT).shift(DOWN * 0.3)

        # Labels for energy levels
        label_5S = Tex(r"$\left|5S\right\rangle$", color=BLACK).next_to(level_5S, LEFT)
        label_5P = Tex(r"$\left|5P\right\rangle$", color=BLACK).next_to(level_5P, RIGHT)
        label_5D = Tex(r"$\left|5D\right\rangle$", color=BLACK).next_to(level_5D, RIGHT)
        label_100S = Tex(r"$\left|100S\right\rangle$", color=BLACK).next_to(level_100S, LEFT)
        label_99P = Tex(r"$\left|99P\right\rangle$", color=BLACK).next_to(level_99P, RIGHT)

        # Arrows
        d = 0.4
        omega_780 = Arrow(start=level_5S.get_top(), end=level_5P.get_bottom() + d * RIGHT, color=RED).add_updater(
            lambda x: x.become(Arrow(start=level_5S.get_top(), end=level_5P.get_bottom() + d * LEFT, color=RED)))


        omega_776 = Arrow(start=level_5P.get_top() + d * LEFT, end=level_5D.get_bottom(), color=PURPLE)

        omega_480 = Arrow(start=level_5P.get_top() + d * LEFT, end=level_100S.get_bottom(), color=BLUE).add_updater(
            lambda x: x.become(Arrow(start=level_5P.get_top() + d * LEFT, end=level_100S.get_bottom(), color=BLUE)))


        omega_1260 = Arrow(start=level_5D.get_top(), end=level_99P.get_bottom(), color=GREEN).add_updater(
            lambda x: x.become(Arrow(start=level_5D.get_top(), end=level_99P.get_bottom(), color=GREEN)))

        # Transition labels
        label_omega_780 = Tex(r"$\Omega_{780}$", color=BLACK).next_to(omega_780, RIGHT)
        label_omega_776 = Tex(r"$\Omega_{776}$", color=BLACK).next_to(omega_776, RIGHT)
        label_omega_480 = Tex(r"$\Omega_{480}$", color=BLACK).next_to(omega_480, LEFT)#.shift(1 * RIGHT + 0.25 * DOWN)
        label_omega_1260 = Tex(r"$\Omega_{1260}$", color=BLACK).next_to(omega_1260, RIGHT)

        # Axes for graphs
        axes = Axes(
            x_range=[-80, 80, 10],
            y_range=[0, 1.1],
            axis_config={"color": BLACK},
            x_length=6,
            y_length=5,
            x_axis_config={"numbers_to_include": [-40, 0, 40]},
        )

        axes.to_edge(RIGHT).shift(DOWN*0.45)

        axes.x_axis.numbers.set_color(BLACK)
        # axes.y_axis.numbers.set_color(BLACK)

        y_axis_1_mark = Tex("1", color=BLACK).move_to(axes.c2p(-10, 1)).scale(0.75)
        y_axis_0p3_mark = Tex("0.3", color=BLACK).move_to(axes.c2p(-10, 0.3)).scale(0.75)




        labels = axes.get_axis_labels(
            Tex(r"$\Delta p$", color=BLACK).scale(0.7), Tex("$T$", color=BLACK).scale(0.7)
        )

        units = Text(r"MHz", color=BLACK).scale(0.5).next_to(axes.x_axis.get_right(), DOWN)

        # Graphs
        graph_1 = axes.plot(lambda x: calculate_transmission(
            x, Gamma,
            Omega_776=0 * MHz, Delta_776=0, gamma_gd=0.06 * MHz,
            Omega_480=0 * MHz, Delta_480=0 * MHz, gamma_gs=0.3 * MHz,
            Omega_1260=0 * MHz, Delta_1260=0 * MHz, gamma_gr=0.3 * MHz,
            OD=OD
        ), color=RED)

        graph_2 = axes.plot(lambda x: calculate_transmission(
            x, Gamma,
            Omega_776=40 * MHz, Delta_776=0, gamma_gd=0.06 * MHz,
            Omega_480=0 * MHz, Delta_480=-40 * MHz, gamma_gs=0.3 * MHz,
            Omega_1260=0 * MHz, Delta_1260=40 * MHz, gamma_gr=0.3 * MHz,
            OD=OD
        ), color=PURPLE)

        graph_3 = axes.plot(lambda x: calculate_transmission(
            x, Gamma,
            Omega_776=40 * MHz, Delta_776=0, gamma_gd=0.06 * MHz,
            Omega_480=5 * MHz, Delta_480=-40 * MHz, gamma_gs=0.3 * MHz,
            Omega_1260=0 * MHz, Delta_1260=40 * MHz, gamma_gr=0.3 * MHz,
            OD=OD
        ), color=BLUE)

        graph_4 = axes.plot(lambda x: calculate_transmission(
            x, Gamma,
            Omega_776=40 * MHz, Delta_776=0, gamma_gd=0.06 * MHz,
            Omega_480=5 * MHz, Delta_480=-40 * MHz, gamma_gs=0.3 * MHz,
            Omega_1260=5 * MHz, Delta_1260=40 * MHz, gamma_gr=0.3 * MHz,
            OD=OD
        ), color=GREEN)

        # Titles
        # title_1 = Tex(r"Transmission with All Parameters Set to Zero").scale(0.75).to_edge(UP).shift(RIGHT * 3)
        # title_2 = Tex(r"Transmission with Non-Zero $\Omega_{776}$").scale(0.75).to_edge(UP).shift(RIGHT * 3)
        # title_3 = Tex(r"Transmission with Non-Zero $\Omega_{480}$").scale(0.75).to_edge(UP).shift(RIGHT * 3)
        # title_4 = Tex(r"Transmission with Non-Zero $\Omega_{1260}$").scale(0.75).to_edge(UP).shift(RIGHT * 3)

        # Step 1: Draw energy levels

        self.play(Create(level_5S), Write(label_5S), Create(level_5P), Write(label_5P), Create(level_5D),
                  Write(label_5D), Create(level_100S), Write(label_100S), Create(level_99P), Write(label_99P),
                  Create(axes), Create(labels), Create(units), Create(y_axis_1_mark), Create(y_axis_0p3_mark))

        # Step 2: Graph 1 with omega_780
        self.next_slide()
        self.play(Create(graph_1), Create(omega_780), Write(label_omega_780))

        # self.wait(2)
        self.next_slide()

        # Step 3: Graph 2 with omega_776
        self.play(
            Transform(graph_1, graph_2),
            Create(omega_776), Write(label_omega_776),
            # FadeOut(title_1, shift=DOWN), FadeIn(title_2, shift=DOWN),
            run_time=2
        )
        # self.wait(2)
        self.next_slide()

        # Step 4: Graph 3 with omega_480
        self.play(
            Transform(graph_1, graph_3),
            Create(omega_480), Write(label_omega_480),
            # FadeOut(title_2, shift=DOWN), FadeIn(title_3, shift=DOWN),
            run_time=2
        )
        # self.wait(2)
        self.next_slide()

        # Step 5: Graph 4 with omega_1260
        self.play(
            Transform(graph_1, graph_4),
            Create(omega_1260), Write(label_omega_1260),
            # FadeOut(title_3, shift=DOWN), FadeIn(title_4, shift=DOWN),
            run_time=2
        )
        # self.wait(2)
        self.next_slide()


        self.play(FadeOut(label_omega_776), FadeOut(omega_776),
                  level_5P.animate.shift(UP * 0.3),
                  level_5D.animate.shift(DOWN * 0.3),
                  label_5P.animate.become(label_minus),
                  label_5D.animate.become(label_plus),
                  )
        omega_780_new = Arrow(start=level_5S.get_top(), end=level_5D.get_bottom() + 1.3 * d * RIGHT, color=RED)
        omega_1260.clear_updaters()
        label_80MHz = Tex(r"$80\text{MHz}$", color=BLACK).move_to((level_5D.get_right() + level_5P.get_right()) / 2).shift(2 * RIGHT).scale(0.7)
        self.play(FadeIn(omega_780_new),
                  omega_1260.animate.become(
                      Arrow(start=level_5D.get_top() + 1.3 * d * RIGHT, end=level_99P.get_bottom(), color=GREEN)),
                  FadeIn(label_80MHz)
                  )


# manim_slides convert Seminar slides/presentation.html