from manim import *
import numpy as np


A = 1
B = np.sqrt(2)


class Torus(ThreeDScene):

    @staticmethod
    def torus(u, v):
        return np.array([
            (2 + np.cos(v)) * np.cos(u),
            (2 + np.cos(v)) * np.sin(u),
            np.sin(v)
        ])

    def construct(self):
        axes = ThreeDAxes(x_range=[-4,4], x_length=8)
        surface = Surface(
            lambda u, v: axes.c2p(*self.torus(u, v)),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=30,
            fill_opacity=0.8
        )
        self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
        self.add(axes, surface)

        line = ParametricFunction(
            lambda t: axes.c2p(*self.torus(t * A, t * B)),
            t_range=[0, 60*np.pi],
            color=PURPLE,
            # stroke_opacity=0.8
        )

        dots = [always_redraw(lambda s=s: Dot3D(color=RED).move_to(axes.c2p(*self.torus((s + self.time) * A, (s + self.time) * B))))
                for s in np.linspace(0, 2, 10)]

        # dot = always_redraw(lambda s=2: Dot3D(color=RED).move_to(axes.c2p(*self.torus((s + self.time) * A, (s + self.time) * B))))

        # dot = always_redraw(lambda t: t.move_to(axes.c2p(*self.torus((3 + self.time) * A, (3 + self.time) * B))))

        dot = Dot3D(color=RED).move_to(axes.c2p(*self.torus((3) * A, (3) * B)))

        def updater_function(object: Mobject):
            object.move_to(axes.c2p(*self.torus((3 + self.time) * A, (3 + self.time) * B)))

        uppdater_function_lambda = lambda object: object.move_to(axes.c2p(*self.torus((3 + self.time) * A, (3 + self.time) * B)))

        dot.add_updater(updater_function)

        self.add(*dots, line)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(2)



class Dummy(Scene):
    def construct(self):
        image_path = r"C:\Users\michaeka\Desktop\260204134449.png"  # <-- change this

        img = ImageMobject(image_path).scale(2)

        # Define a funny, wiggly path using arbitrary points
        path = VMobject()
        path.set_points_smoothly([
            LEFT * 7 + DOWN * 2,
            LEFT * 3 + UP * 2,
            ORIGIN + DOWN * 1,
            RIGHT * 3 + UP * 1.5,
            RIGHT * 6 + DOWN * 2,
        ])

        # Move image to start of path
        img.move_to(path.get_start())

        self.add(img)

        # Animate movement along the path + rotation
        self.play(
            MoveAlongPath(img, path),
            Rotate(
                img,
                angle=6 * TAU,           # lots of spinning
                about_point=img.get_center()
            ),
            run_time=4,
            rate_func=smooth
        )

        self.wait()



t = np.linspace(0, 10, 10000)
x = np.cos(np.sin(t))**2 / 2
y = np.sin(np.cos(t))**2 / 2

def r_t_floored(t_current):
    # Return the index
    relevant_index = np.searchsorted(t, t_current)
    return x[relevant_index], y[relevant_index]



class NumericalMovement(Scene):
    def construct(self):
        dot = Dot(color=RED).move_to((x[0], y[0], 0))
        t_tracker = ValueTracker(0)
        dot.add_updater(lambda d: d.move_to(np.array([r_t_floored(t_tracker.get_value())[0], r_t_floored(t_tracker.get_value())[1], 0])))
        self.add(dot)
        self.play(t_tracker.animate.increment_value(10), run_time=6)