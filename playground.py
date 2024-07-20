from manim import *

class SimpleDiagram(Scene):
    def construct(self):
        green_color = GREEN
        blue_color = BLUE
        purple_color = PURPLE
        red_color = RED

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
            Dot().scale(0.5).move_to((theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2 + vertical_spacing * 0.15 * UP),
            Dot().scale(0.5).move_to((theta_blocks[0].get_center() + theta_blocks[1].get_center()) / 2 + vertical_spacing * 0.15 * DOWN)
        )

        # Lines connecting the blocks
        lines = VGroup(
  Line(preparation_block.get_right() + UP * vertical_spacing, measurement_block.get_left() + UP * vertical_spacing, color=red_color),
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

        multiple_phase_scheme = VGroup(lines, preparation_block, measurement_block, prep_label, meas_label, theta_blocks, theta_labels, dots)
        self.play(FadeIn(multiple_phase_scheme))
        self.play(FadeOut(multiple_phase_scheme))



if __name__ == "__main__":
    from manim import *
    config.media_width = "100%"
    scene = SimpleDiagram()
    scene.render()