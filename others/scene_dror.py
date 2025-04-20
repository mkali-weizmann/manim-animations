from manim import *
# manim -pqh slides/scene_dror.py Licakge
ARC_ANGLE = 0.8
ARC_CENTER = 0.3
PHOTON_WIDTH = 0.09
ARC_RADIUS = 3
BEAM_SPLITTER_WIDTH = 0.3
BEAM_SPLITTER_STROKE_WIDTH = 2.1
PHOTON_LEFT_LEFT_LIMIT = ARC_CENTER-ARC_RADIUS+PHOTON_WIDTH
PHOTON_LEFT_RIGHT_LIMIT = (-BEAM_SPLITTER_WIDTH) / 2

PHOTON_RIGHT_LEFT_LIMIT = (BEAM_SPLITTER_WIDTH) / 2
PHOTON_RIGHT_RIGHT_LIMIT = -ARC_CENTER+ARC_RADIUS-PHOTON_WIDTH
BOMB_POSITION = (-ARC_CENTER+ARC_RADIUS) / 2

t = ValueTracker(0)
V_1 = 0.8


def zig_zag_function(x, a, b):
    return (b-a) * 2 * np.abs(np.mod(x-0.5,1) -0.5) + a


def steps_function(x, a):
    return a * np.floor(x)


class Licakge(Scene):
    def construct(self):
        arc_left = Arc(radius=ARC_RADIUS, start_angle=PI-ARC_ANGLE/2, angle=ARC_ANGLE, arc_center=[ARC_CENTER, 0, 0])
        arc_2_right = Arc(radius=ARC_RADIUS, start_angle=-ARC_ANGLE/2, angle=ARC_ANGLE, arc_center=[-ARC_CENTER, 0, 0])
        beam_splitter = Rectangle(height=1, width=BEAM_SPLITTER_WIDTH, color=BLUE, stroke_width=0.7)
        photon_left = Dot(color=RED, radius=PHOTON_WIDTH).move_to([ARC_CENTER-ARC_RADIUS+PHOTON_WIDTH, 0, 0])
        photon_left.add_updater(lambda m: m.move_to([zig_zag_function(t.get_value(), PHOTON_LEFT_LEFT_LIMIT, PHOTON_LEFT_RIGHT_LIMIT), 0, 0]))
        photon_right = Dot(color=RED, radius=PHOTON_WIDTH).move_to([-ARC_CENTER+ARC_RADIUS-PHOTON_WIDTH, 0, 0])
        photon_right.add_updater(lambda m: m.move_to([zig_zag_function(t.get_value(), PHOTON_RIGHT_RIGHT_LIMIT, PHOTON_RIGHT_LEFT_LIMIT), 0, 0]).set_opacity(steps_function(t.get_value()+0.5, 0.2)))
        # load "slides\bomb.svg" as an svg object:
        bomb = SVGMobject(r"slides\bomb.svg").scale(0.2).move_to([(ARC_RADIUS-ARC_CENTER) / 2, 0, 0]).set_opacity(0.5)
        self.add(arc_left, arc_2_right, beam_splitter, photon_left, photon_right, bomb)

        self.play(t.animate.increment_value(1.5), run_time=1.5/V_1, rate_func=linear)
        self.play(Wiggle(bomb, scale_value=1.4), t.animate.increment_value(1), run_time=1/V_1, rate_func=linear)
        # self.play(t.animate.increment_value(0.5), run_time=0.5/V_1, rate_func=linear)


class NoBomb(Scene):
    def construct(self):
        arc_left = Arc(radius=ARC_RADIUS, start_angle=PI-ARC_ANGLE/2, angle=ARC_ANGLE, arc_center=[ARC_CENTER, 0, 0])
        arc_2_right = Arc(radius=ARC_RADIUS, start_angle=-ARC_ANGLE/2, angle=ARC_ANGLE, arc_center=[-ARC_CENTER, 0, 0])
        beam_splitter = Rectangle(height=1, width=BEAM_SPLITTER_WIDTH, color=BLUE, stroke_width=0.7)
        photon_left = Dot(color=RED, radius=PHOTON_WIDTH).move_to([ARC_CENTER-ARC_RADIUS+PHOTON_WIDTH, 0, 0])
        photon_left.add_updater(lambda m: m.move_to([zig_zag_function(t.get_value(), PHOTON_LEFT_LEFT_LIMIT, PHOTON_LEFT_RIGHT_LIMIT), 0, 0]))
        # load "slides\bomb.svg" as an svg object:
        bomb = SVGMobject(r"slides\bomb.svg").scale(0.2).move_to([(ARC_RADIUS-ARC_CENTER) / 2, 0, 0]).set_opacity(0.5)
        self.add(arc_left, arc_2_right, beam_splitter, photon_left, bomb)

        self.play(t.animate.increment_value(3), run_time=3/V_1, rate_func=linear)
