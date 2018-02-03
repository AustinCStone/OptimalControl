
import gi
gi.require_version('Gtk', '3.0') 
from gi.repository import Gtk, GLib
import cairo
import numpy as np


def draw_rect(cr, rect, rgb):

    cr.set_source_rgb(*rgb)
    pts = [(rect[0], rect[1]), (rect[0] + rect[2], rect[1]), 
           (rect[0] + rect[2], rect[1] + rect[3]), (rect[0], rect[1] + rect[3]),
           (rect[0], rect[1])]

    for i in range(len(pts)):
        cr.line_to(pts[i][0], pts[i][1])
    cr.fill()


class CartPole(Gtk.Window):

    m = 1.;
    M = 5.; # mass of cart
    L = 2.; # length of pendulum
    g = -10.; # accel due to gravity
    d = 1.;

    # simulation discretization
    dt = .01

    # state of the system - position, velocity, angle, angular velocity
    state = [0., 0., np.pi, .5];

    SPEED = 3
    SCALING = 50
    TIMER_ID = 1

    def __init__(self):
        super(CartPole, self).__init__()
        
        self.init_ui()
        self.init_vars()
        
        
    def init_ui(self):    

        self.darea = Gtk.DrawingArea()
        self.darea.set_size_request(1000,1000)
        self.darea.connect("draw", self.on_draw)
        self.add(self.darea)

        self.set_title("CartPole")
        self.resize(200, 200)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()

    def init_vars(self):   

        GLib.timeout_add(self.SPEED, self.on_timer)   
         
    def update_pendulum_dynamics(self):

        u = 0. # TODO: add control
        M = self.M; m = self.m; L = self.L; g = self.g; d = self.d
        y = self.state

        Sy = np.sin(y[2])
        Cy = np.cos(y[2])
        D = m*L*L*(M+m*(1-Cy**2));

        # some magic differential equations describing the evolution of the system
        d_p = y[1]
        d_v = (1/D)*(-m**2*L**2*g*Cy*Sy + m*L**2*(m*L*y[3]**2*Sy - d*y[1])) + m*L*L*(1/D)*u
        d_a = y[3]
        r = np.random.normal()
        d_av = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*y[3]**2*Sy - d*y[1])) - m*L*Cy*(1/D)*u +.01*r

        self.state[0] += d_p * self.dt
        self.state[1] += d_v * self.dt
        self.state[2] += d_a * self.dt
        self.state[3] += d_av * self.dt

    def on_timer(self):
        
        self.update_pendulum_dynamics()
    
        self.darea.queue_draw()
        
        return True 

    def on_draw(self, wid, cr):

        w, h = self.get_size()

        cr.translate(w/2, h/2)
        cr.rotate(np.pi)
        cr.scale(self.SCALING, self.SCALING)

        x = self.state[0]
        th = self.state[2]

        M = self.M; m = self.m; L = self.L; g = self.g; d = self.d

        # kinematics
        W = 1*np.sqrt(M/5)  # cart width
        H = .5*np.sqrt(M/5) # cart height
        wr = .2 # wheel radius
        mr = .3*np.sqrt(m) # mass radius

        # positions
        y = wr/2+H/2 # cart vertical position
        w1x = x-.9*W/2
        w1y = 0
        w2x = x+.9*W/2-wr
        w2y = 0

        px = x + L*np.sin(th);
        py = y - L*np.cos(th);

        draw_rect(cr, [x-W/2,y-H/2,W,H], [1., 0., 0.])
        draw_rect(cr, [w1x,w1y,wr,wr], [0., 1., 0.])
        draw_rect(cr, [w2x,w2y,wr,wr], [0., 1., 0.])
        draw_rect(cr, [px-mr/2,py-mr/2,mr,mr], [0., 0., 1.])
        cr.set_line_width(.01)
        cr.move_to(x, y)
        cr.line_to(px, py)
        cr.stroke()
        cr.fill()
        
def main():
    
    app = CartPole()
    Gtk.main()
        
        
if __name__ == "__main__":    
    main()