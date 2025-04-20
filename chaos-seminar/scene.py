from manim import *
import copy

def tent(x, n=1):
    if n==0:
        return x
    else:
        for i in range(n):
            x = 1-2*abs(x-1/2)
        return x


def twitched_tent(x, n=1):
    y = tent(x, n)
    if y == 0:
        return np.random.uniform()
    else:
        return y


def LogisticMap(x, r, n=1):
    if n==0:
        return x
    else:
        for i in range(n):
            x = r*x*(1-x)
        return x


def modulouMap(x, n=1):
    if n==0:
        return x
    else:
        for i in range(n):
            x = (2*x)%1
        return x


def strech(x):
    return 2*x
  
  
def identity_func(x):
    return x

###########################


class ModulouNSimplePlot(GraphScene, Scene):
    

    
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0,
            y_axis_label=r'$M^{3}\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_max=1.1,
            x_axis_width=9,
            y_axis_height=5.2,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 4.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 1.1),
            x_axis_config={"tick_frequency": 1},
            x_labeled_nums=np.arange(0, 1, 1),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.005
        graph_x_max = 0.99
        dt=1e-3
        epsilon_x = 0.001
        modulou_color = RED
        identity_color = TEAL_B
        
        l = np.linspace(0, 1, 9)
        intersection_points = np.linspace(0, 6/7, 7)
        
        tent_graph_1 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[0], x_max=l[1]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_2 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[1], x_max=l[2]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_3 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[2], x_max=l[3]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_4 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[3], x_max=l[4]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_5 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[4], x_max=l[5]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_6 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[5], x_max=l[6]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_7 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[6], x_max=l[7]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_8 = self.get_graph(lambda x: modulouMap(x, 3), x_min=l[7], x_max=l[8]-epsilon_x, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        tent_graph_9 = Dot(point=self.coords_to_point(1, 0), radius=0.05).set_color(modulou_color)
        
        dot_0 = AnnotationDot(point=self.coords_to_point(intersection_points[0], intersection_points[0]), radius=0.08)
        dot_1 = AnnotationDot(point=self.coords_to_point(intersection_points[1], intersection_points[1]), radius=0.08)
        dot_2 = AnnotationDot(point=self.coords_to_point(intersection_points[2], intersection_points[2]), radius=0.08)
        dot_3 = AnnotationDot(point=self.coords_to_point(intersection_points[3], intersection_points[3]), radius=0.08)
        dot_4 = AnnotationDot(point=self.coords_to_point(intersection_points[4], intersection_points[4]), radius=0.08)
        dot_5 = AnnotationDot(point=self.coords_to_point(intersection_points[5], intersection_points[5]), radius=0.08)
        dot_6 = AnnotationDot(point=self.coords_to_point(intersection_points[6], intersection_points[6]), radius=0.08)
        
        
        
        eq_title = MathTex(r'M^{3}\left(x_{n}\right)=\left(2^{3}x_{n}\right)\text{mod}1').to_corner(UP+RIGHT).scale(0.8).set_color(modulou_color)
        
        self.play(Create(tent_graph_1), Create(tent_graph_2), Create(tent_graph_3), Create(tent_graph_4), Create(tent_graph_5), Create(tent_graph_6), Create(tent_graph_7), Create(tent_graph_8), Create(tent_graph_9), Write(eq_title))
        
        identity_graph = self.get_graph(identity_func, x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(identity_color)
        identity_title = MathTex(r'y=x').set_color(identity_color).next_to(eq_title, DOWN)
        self.play(Create(identity_graph), Create(identity_title))
        self.play(Create(dot_0), Create(dot_1), Create(dot_2), Create(dot_3), Create(dot_4), Create(dot_5), Create(dot_6))


class ModulouSimplePlot(GraphScene, Scene):
    

    
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=2.2,
            x_min=0,
            y_axis_label=r'$M\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_max=1.1,
            x_axis_width=5,
            y_axis_height=5.2,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 1},
            x_labeled_nums=np.arange(0, 1, 1),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.01
        graph_x_max = 0.99
        dt=8e-3
        modulou_color = RED
        
        
        main_graph = self.get_graph(identity_func, x_min=0, x_max=graph_x_max, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(modulou_color)
        stretch_graph = self.get_graph(strech, x_min=0, x_max=graph_x_max, y_min=0, y_max=1, t_range=[0, 1, resolution], discontinuities=[0.5]).set_color(modulou_color)
        tent_graph = self.get_graph(lambda x: modulouMap(x, 1), x_min=0, x_max=graph_x_max, y_min=0, y_max=1, t_range=[0, 1, resolution], discontinuities=[0.5], use_smoothing=False, dt=dt).set_color(modulou_color)
        
        eq_title = MathTex(r'M\left(x_{n}\right)=\left(2x_{n}\right)\text{mod}1').shift(title_position+0.4*RIGHT).scale(0.8).set_color(modulou_color)
        stretch_title = Tex(r'Strech:').shift(title_position)
        fold_title= Tex(r'Fold:').shift(title_position)
        

        self.play(Create(main_graph))
        self.play(Write(stretch_title))
        self.play(Transform(main_graph, stretch_graph))

        self.play(FadeOut(stretch_title),FadeInFrom(fold_title, direction=DOWN))
        
        self.play(Transform(main_graph, tent_graph))
        
        self.wait()
        
        self.play(FadeOut(fold_title))
        
        self.play(Write(eq_title))


class TentSimplePlot(GraphScene, Scene):
    

    
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=2.2,
            x_min=0,
            y_axis_label=r'$M\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_max=1.1,
            x_axis_width=5,
            y_axis_height=5.2,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 1},
            x_labeled_nums=np.arange(0, 1, 1),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.01
        
        main_graph = self.get_graph(identity_func, x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(BLUE)
        stretch_graph = self.get_graph(strech, x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(BLUE)
        tent_graph = self.get_graph(lambda x: tent(x, 1), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(BLUE)

        eq_title = MathTex(r'{\color{blue}M\left(x_{n}\right)}=1-2\left|x_{n}-\frac{1}{2}\right|').shift(title_position+0.4*RIGHT+0.4*DOWN).scale(0.8).set_color(BLUE)
        stretch_title = Tex(r'Strech:').shift(title_position)
        fold_title= Tex(r'Fold:').shift(title_position)

        self.play(Create(main_graph))
        self.play(Write(stretch_title))
        self.play(Transform(main_graph, stretch_graph))

        self.play(FadeOut(stretch_title),FadeInFrom(fold_title, direction=DOWN))
        
        self.play(Transform(main_graph, tent_graph))
        
        self.wait()
        
        self.play(FadeOut(fold_title))
        
        self.play(Write(eq_title))


class TentInterationPlot(GraphScene, Scene):
    

    
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=2.2,
            x_min=0,
            x_max=1.1,
            x_axis_width=5,
            y_axis_height=5.2,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 1},
            x_labeled_nums=np.arange(0, 1, 1),
            **kwargs
        )

    def construct(self):
    
        m = 4
        resolution = 0.0001
        title_position = 3*UP + 3.5*RIGHT
    
        self.setup_axes()

        title = MathTex(r'M^{\bold{%d}}\left(x\right)' % 0)
        title.shift(title_position).set_color(BLUE)
        self.add(title)
        main_graph = self.get_graph(identity_func, x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(BLUE)
        self.add(main_graph)
        self.wait()

        
        for i in range(1, m+1):
            updated_title = MathTex(r'M^{\bold{%d}}\left(x\right)' % i)
            updated_title.shift(title_position).set_color(BLUE)

            temp_graph = self.get_graph(lambda x: strech(tent(x, i)), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(BLUE)
            self.play(Transform(main_graph, temp_graph))
            
            temp_graph = self.get_graph(lambda x: tent(x, i+1), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(BLUE)
            self.play(Transform(main_graph, temp_graph), Transform(title, updated_title))
        

class LogisticMapFirstBifurcation(GraphScene, MovingCameraScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0,
            x_max=1.1,
            x_axis_width=8,
            y_axis_height=5.5,
            y_label_position=np.array([-1.0, 1.0, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 1),
            x_axis_config={"tick_frequency": 1},
            x_labeled_nums=np.arange(0, 1, 1),
            **kwargs
        )


    
    def construct(self):
        self.setup_axes()
        
        title_position = 3*UP + 4.5*RIGHT
        resolution = 0.01
        M1_color = GREEN
        M2_color = BLUE
        identity_color = RED
        zoom_ratio_tracker = ValueTracker(1)
        points_radius = 0.02
        
        r_tracker = ValueTracker(2.8)
        
        L1_graph = self.get_graph(lambda x: LogisticMap(x, r_tracker.get_value(), 1), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(M1_color).set_stroke(width=1)
        L2_graph = self.get_graph(lambda x: LogisticMap(x, r_tracker.get_value(), 2), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(M2_color).set_stroke(width=1)
        identity_graph = self.get_graph(identity_func, x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(identity_color).set_stroke(width=1)
        first_intersection_point = Dot(radius=points_radius*1.3, color=YELLOW).move_to(self.coords_to_point(1-1/r_tracker.get_value(), 1-1/r_tracker.get_value()))
        
        M1_legend = MathTex(r'M\left(x\right)').set_color(M1_color).move_to(title_position)
        M2_legend = MathTex(r'M^{2}\left(x\right)').set_color(M2_color).move_to(M1_legend.get_center()+0.6*DOWN)
        identity__legend = MathTex(r'y=x').set_color(identity_color).move_to(M1_legend.get_center()+0.6*DOWN)
        r_legend = MathTex(r'r=%.3f' % r_tracker.get_value()).move_to(M2_legend.get_center()+0.6*DOWN)

        
        self.play(Create(L1_graph), Create(L2_graph), Create(identity_graph), Create(M1_legend), Create(M2_legend), Create(r_legend))
        self.play(Create(first_intersection_point))


        L1_graph.add_updater(
            lambda x: x.become(self.get_graph(lambda x: LogisticMap(x, r_tracker.get_value(), 1), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(M1_color)).set_stroke(width=0.8)
        )
        L2_graph.add_updater(
            lambda x: x.become(self.get_graph(lambda x: LogisticMap(x, r_tracker.get_value(), 2), x_min=0, x_max=1, y_min=0, y_max=1, t_range=[0, 1, resolution]).set_color(M2_color)).set_stroke(width=0.8)
        )
        
        r_legend.add_updater(
            lambda x: x.become(MathTex(r'r=%.3f' % r_tracker.get_value()).move_to(M2_legend.get_center()+0.6*zoom_ratio_tracker.get_value()*DOWN)).scale(zoom_ratio_tracker.get_value())
        )
        
        
#        M2_legend.add_updater(
#            lambda x: x.move_to(M1_legend.get_center()+0.6*zoom_ratio_tracker.get_value()*DOWN)#.scale(zoom_ratio_tracker.get_value())
#        )
        
        # M1_legend.add_updater(
            # lambda x: x.scale(zoom_ratio_tracker.get_value())
        # )
        
        first_intersection_point.add_updater(
            lambda x: x.move_to(self.coords_to_point(1-1/r_tracker.get_value(), 1-1/r_tracker.get_value()))
        )
        
        
        # second_intersection_points_2.add_updater(
            # lambda x: x.move_to(self.coords_to_point((1+(r_tracker.get_value())+((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5)/(2*(r_tracker.get_value())), (1+(r_tracker.get_value())+((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5)/(2*(r_tracker.get_value()))))
        # )
        
        
        self.play(r_tracker.animate.set_value(2.95), run_time=4)
        zoom_ratio = 0.3
        self.play(zoom_ratio_tracker.animate.set_value(zoom_ratio),
                  self.camera.frame.animate.scale(zoom_ratio).move_to(first_intersection_point),
                  M1_legend.animate.move_to(first_intersection_point.get_center()+UP).scale(zoom_ratio),
                  M2_legend.animate.move_to(first_intersection_point.get_center()+UP+DOWN*0.6*zoom_ratio).scale(zoom_ratio)#,
                  #r_legend.animate.move_to(first_intersection_point).scale(zoom_ratio)
                  )
        self.play(r_tracker.animate.set_value(3), run_time=6)
        print(r_tracker.get_value())
        
        second_intersection_points_1 = Dot(radius=points_radius, color=ORANGE).move_to(self.coords_to_point((1+(r_tracker.get_value())-((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value())),(1+(r_tracker.get_value())-((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value()))))
        second_intersection_points_2 = Dot(radius=points_radius, color=ORANGE).move_to(self.coords_to_point((1+(r_tracker.get_value())+((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value())),(1+(r_tracker.get_value())+((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value()))))
        second_intersection_points_1.add_updater(
            lambda x: x.move_to(self.coords_to_point((1+(r_tracker.get_value())-((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value())),(1+(r_tracker.get_value())-((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value()))))
        )
        second_intersection_points_2.add_updater(
            lambda x: x.move_to(self.coords_to_point((1+(r_tracker.get_value())+((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value())),(1+(r_tracker.get_value())+((r_tracker.get_value())**2-2*(r_tracker.get_value())-3)**0.5) / (2*(r_tracker.get_value()))))
        )
        
        self.play(Create(second_intersection_points_1), Create(second_intersection_points_2))
        self.play(r_tracker.animate.set_value(3.15), run_time=6)

            
class HistogramChart(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                y_min=0,
                y_max=1.1,
                x_min=0,
                x_max=1.1,
                x_axis_width=8,
                y_axis_height=5.5,
                graph_origin=np.array([- 2.5, - 2.5, 0.0]),
                y_label_position=np.array([-1.0, 1.4, 0.0]),
                #y_axis_config={"tick_frequency": 1},
                #y_labeled_nums=np.arange(0, 1, 1),
                #x_axis_config={"tick_frequency": 1},
                #x_labeled_nums=np.arange(0, 1, 1),

                **kwargs
            )
    
    def construct(self):
    
       self.setup_axes()
       
       n = 10
       charts_values = np.zeros(n)
       charts_heights = np.zeros(n)
       x = 2**(-0.5)
       
       
       chart = BarChart(values=charts_heights, width = self.x_axis_width, height=self.y_axis_height).move_to(self.graph_origin+2.548*UP + 3.57*RIGHT)
       tent_graph = self.get_graph(lambda x: tent(x, 1), x_min=0, x_max=1, y_min=0, y_max=1).set_color(BLUE)
       dot = Dot(point=self.coords_to_point(x, 0), radius = 0.08)
       self.add(tent_graph, chart)
       
       for i in range(100):
            current_bin = int(np.floor((n-1)*x))
            charts_values[current_bin] +=1
            charts_heights = charts_values / sum(charts_values)
            x = tent(x, 1)
            updatedChart = BarChart(values=charts_heights, width = self.x_axis_width, height=self.y_axis_height).move_to(self.graph_origin+2.548*UP + 3.57*RIGHT)
            self.play(dot.animate.move_to(self.coords_to_point(x, 0)), Transform(chart, updatedChart), run_time=0.1)
       

class TwitchedHistogramChart(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                y_min=0,
                y_max=1.1,
                x_min=0,
                x_max=1.1,
                x_axis_width=8,
                y_axis_height=5.5,
                graph_origin=np.array([- 2.5, - 2.5, 0.0]),
                y_label_position=np.array([-1.0, 1.4, 0.0]),
                #y_axis_config={"tick_frequency": 1},
                #y_labeled_nums=np.arange(0, 1, 1),
                #x_axis_config={"tick_frequency": 1},
                #x_labeled_nums=np.arange(0, 1, 1),

                **kwargs
            )
    
    def construct(self):
    
       self.setup_axes()
       
       n = 10
       charts_values = np.zeros(n)
       charts_heights = np.zeros(n)
       x = 2**(-0.5)
       
       
       chart = BarChart(values=charts_heights, width = self.x_axis_width, height=self.y_axis_height).move_to(self.graph_origin+2.548*UP + 3.57*RIGHT)
       tent_graph = self.get_graph(lambda x: tent(x, 1), x_min=0, x_max=1, y_min=0, y_max=1).set_color(BLUE)
       dot = Dot(point=self.coords_to_point(x, 0), radius = 0.08)
       self.add(tent_graph, chart)
       
       
       for i in range(200):
            current_bin = int(np.floor(n*x))
            if current_bin == n:
               current_bin -=1 
            charts_values[current_bin] +=1
            charts_heights = charts_values / sum(charts_values)
            x = twitched_tent(x, 1)
            #print(charts_heights)
            #print(int(np.floor((n-1)*x)), (n-1)*x, x)
            updatedChart = BarChart(values=charts_heights, width = self.x_axis_width, height=self.y_axis_height).move_to(self.graph_origin+2.548*UP + 3.57*RIGHT)
            self.play(dot.animate.move_to(self.coords_to_point(x, 0)), Transform(chart, updatedChart), run_time=0.1)


class TentZoomIn_3(GraphScene, MovingCameraScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0.53-2**-3,
            x_max=0.53+2**-2,
            y_axis_label=r'$M^{3}\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_axis_width=8,
            y_axis_height=5.5,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 0.006, "x_label_decimal": -2},
            #x_labeled_nums=np.linspace(0.53-2**-5, 0.53+2**-4, 10),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.001
        
        x_0 =0.53
        n = 3
        
        res = 2**-(n-1)
        x_rounded = x_0 - (x_0 % res)
        
        
        tent_left_title = MathTex(r'j*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded, 0), 1.6*DOWN).scale(0.7)
        tent_right_title = MathTex(r'\left(j+1\right)*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded+res, 0), 1.6*DOWN).scale(0.7)
        x_0_title = MathTex(r'x_{0}').next_to(self.coords_to_point(x_0, 0), 0.6*DOWN).scale(0.7)
        
        tent_graph = self.get_graph(lambda x: tent(x, n), x_min=0.53-2**-n, x_max=0.53+2**-(n-1), y_min=0, y_max=1, t_range=[0, 1, resolution], use_smoothing=False).set_color(BLUE)
        vertical_line = self.get_vertical_line_to_graph(x_0, tent_graph, DashedLine, color=YELLOW)

        self.add(tent_graph, vertical_line, x_0_title, tent_left_title, tent_right_title)


class TentZoomIn_4(GraphScene, MovingCameraScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0.53-2**-4,
            x_max=0.53+2**-3,
            y_axis_label=r'$M^{4}\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_axis_width=8,
            y_axis_height=5.5,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 0.006, "x_label_decimal": -2},
            #x_labeled_nums=np.linspace(0.53-2**-5, 0.53+2**-4, 10),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        resolution = 0.001
        
        x_0 =0.53
        n = 4
        
        res = 2**-(n-1)
        x_rounded = x_0 - (x_0 % res)
        
        
        tent_left_title = MathTex(r'j*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded, 0), 1.6*DOWN).scale(0.7)
        tent_right_title = MathTex(r'\left(j+1\right)*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded+res, 0), 1.6*DOWN).scale(0.7)
        x_0_title = MathTex(r'x_{0}').next_to(self.coords_to_point(x_0, 0), 0.6*DOWN).scale(0.7)
        
        tent_graph = self.get_graph(lambda x: tent(x, n), x_min=0.53-2**-n, x_max=0.53+2**-(n-1), y_min=0, y_max=1, t_range=[0, 1, resolution], use_smoothing=False).set_color(BLUE)
        vertical_line = self.get_vertical_line_to_graph(x_0, tent_graph, DashedLine, color=YELLOW)

        self.add(tent_graph, vertical_line, x_0_title, tent_left_title, tent_right_title)


class TentZoomIn_5(GraphScene, MovingCameraScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0.53-2**-5,
            x_max=0.53+2**-4,
            y_axis_label=r'$M^{5}\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_axis_width=8,
            y_axis_height=5.5,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 0.006, "x_label_decimal": -2},
            #x_labeled_nums=np.linspace(0.53-2**-5, 0.53+2**-4, 10),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.001
        
        x_0 =0.53
        n = 5
        
        res = 2**-(n-1)
        x_rounded = x_0 - (x_0 % res)
        
        
        tent_left_title = MathTex(r'j*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded, 0), 1.6*DOWN).scale(0.7)
        tent_right_title = MathTex(r'\left(j+1\right)*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded+res, 0), 1.6*DOWN).scale(0.7)
        x_0_title = MathTex(r'x_{0}').next_to(self.coords_to_point(x_0, 0), 0.6*DOWN).scale(0.7)
        
        tent_graph = self.get_graph(lambda x: tent(x, n), x_min=0.53-2**-n, x_max=0.53+2**-(n-1), y_min=0, y_max=1, t_range=[0, 1, resolution], use_smoothing=False).set_color(BLUE)
        vertical_line = self.get_vertical_line_to_graph(x_0, tent_graph, DashedLine, color=YELLOW)

        self.add(tent_graph, vertical_line, x_0_title, tent_left_title, tent_right_title)
 
 
class TentZoomIn_6(GraphScene, MovingCameraScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0.53-2**-6,
            x_max=0.53+2**-5,
            y_axis_label=r'$M^{6}\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_axis_width=8,
            y_axis_height=5.5,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 0.006, "x_label_decimal": -2},
            #x_labeled_nums=np.linspace(0.53-2**-5, 0.53+2**-4, 10),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.001
        
        x_0 =0.53
        n = 6
        
        res = 2**-(n-1)
        x_rounded = x_0 - (x_0 % res)
        
        
        tent_left_title = MathTex(r'j*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded, 0), 1.6*DOWN).scale(0.7)
        tent_right_title = MathTex(r'\left(j+1\right)*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded+res, 0), 1.6*DOWN).scale(0.7)
        x_0_title = MathTex(r'x_{0}').next_to(self.coords_to_point(x_0, 0), 0.6*DOWN).scale(0.7)
        
        tent_graph = self.get_graph(lambda x: tent(x, n), x_min=0.53-2**-n, x_max=0.53+2**-(n-1), y_min=0, y_max=1, t_range=[0, 1, resolution], use_smoothing=False).set_color(BLUE)
        vertical_line = self.get_vertical_line_to_graph(x_0, tent_graph, DashedLine, color=YELLOW)

        self.add(tent_graph, vertical_line, x_0_title, tent_left_title, tent_right_title)


class TentZoomIn_7(GraphScene, MovingCameraScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0.53-2**-7,
            x_max=0.53+2**-6,
            y_axis_label=r'$M^{7}\left(x_{n}\right)$',
            x_axis_label=r'$x_{n}$',
            x_axis_width=8,
            y_axis_height=5.5,
            y_label_position=np.array([0, 1.0, 0.0]),
            graph_origin=np.array([- 2.5, - 2.5, 0.0]),
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 2),
            x_axis_config={"tick_frequency": 0.006, "x_label_decimal": -2},
            #x_labeled_nums=np.linspace(0.53-2**-5, 0.53+2**-4, 10),
            **kwargs
        )

    def construct(self):
    
    
        self.setup_axes()
        title_position = 3*UP
        resolution = 0.001
        
        x_0 =0.53
        n = 7
        
        res = 2**-(n-1)
        x_rounded = x_0 - (x_0 % res)
        
        
        tent_left_title = MathTex(r'j*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded, 0), 1.6*DOWN).scale(0.7)
        tent_right_title = MathTex(r'\left(j+1\right)*2^{-%d+1}' % n).next_to(self.coords_to_point(x_rounded+res, 0), 1.6*DOWN).scale(0.7)
        x_0_title = MathTex(r'x_{0}').next_to(self.coords_to_point(x_0, 0), 0.6*DOWN).scale(0.7)
        
        tent_graph = self.get_graph(lambda x: tent(x, n), x_min=0.53-2**-n, x_max=0.53+2**-(n-1), y_min=0, y_max=1, t_range=[0, 1, resolution], use_smoothing=False).set_color(BLUE)
        vertical_line = self.get_vertical_line_to_graph(x_0, tent_graph, DashedLine, color=YELLOW)

        self.add(tent_graph, vertical_line, x_0_title, tent_left_title, tent_right_title)


class TentToLogistic(GraphScene):

        
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=1.1,
            x_min=0,
            x_max=1.1,
            x_axis_width=4,
            y_axis_height=4,
            y_axis_config={"tick_frequency": 1},
            y_labeled_nums=np.arange(0, 1, 1),
            x_axis_config={"tick_frequency": 1},
            x_labeled_nums=np.arange(0, 1, 1),
            **kwargs
        )


    
    def construct(self):
            
        self.setup_axes()
        # t = ValueTracker(0)
        L_graph = self.get_graph(lambda x: LogisticMap(x, 4, 1), x_min=0, x_max=1, y_min=0, y_max=1).set_color(BLUE).set_stroke(width=4)
        T_graph = self.get_graph(lambda x: tent(x, 1), x_min=0, x_max=1, y_min=0, y_max=1).set_color(BLUE).set_stroke(width=4)
        
        # n = 11   
        # xs = np.linspace(0, 1, n)
        # dots = []
        # for i in range(n):
            # dots.append(Dot(radius = 0.2).move_to(self.coords_to_point(copy.copy(xs[i]), 0)))
            #dots[-1].add_updater(
             #   lambda d: d.move_to(self.coords_to_point( (1-t.get_value()) * copy.copy(xs[i]) + t.get_value() * (np.sin(PI * copy.copy(xs[i]) / 2))**2, 0) )
            #)
        
        # d1 = Dot(radius = 0.2).move_to(self.coords_to_point(0.1, 0))
        # d1.add_updater(
                 # lambda d: d.move_to(self.coords_to_point( (1-t.get_value()) * 0.1 + t.get_value() * (np.sin(PI * 0.1 / 2))**2, 0) )
            # )

        
        self.add(T_graph)# , *dots
        self.wait()
        self.play(Transform(T_graph, L_graph) )# , t.animate.set_value(1)*[d.animate.move_to(  self.coords_to_point((np.sin(PI * copy.copy(xs[i]) / 2))**2, 0)   )]

        
        
        
        
        
        
        