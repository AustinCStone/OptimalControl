
import gi
gi.require_version('Gtk', '3.0') 
from gi.repository import Gtk, GLib, Gdk
import cairo
import copy
import numpy as np
import scipy.linalg


def draw_rect(cr, rect, rgb):
    """
    Draw a rectangle with cairo. rect is yxhw format
    """
    cr.set_source_rgb(*rgb)
    pts = [(rect[0], rect[1]), (rect[0] + rect[2], rect[1]), 
           (rect[0] + rect[2], rect[1] + rect[3]), (rect[0], rect[1] + rect[3]),
           (rect[0], rect[1])]

    for i in range(len(pts)):
        cr.line_to(pts[i][0], pts[i][1])
    cr.fill()

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals

class CartPole(Gtk.Window):
    """
    Class for rendering and controlling
    an inverted pendulum on a cart.
    """

    m = 50.; # mass of the top knob on the pendulum
    M = 5.; # mass of cart
    L = 2.; # length of pendulum
    g = -10.; # accel due to gravity
    d = 1.;

    # simulation discretization
    dt = .1

    # state of the system - position, velocity, angle, angular velocity
    # initial state is with the pendulum pointing up
    state = np.asarray([8., 0., np.pi + .2, 0.])
    desired_x = 3.
    SPEED = 3
    SCALING = 50
    TIMER_ID = 1

    def __init__(self, use_lqr_control=True):
        super(CartPole, self).__init__()
        
        self.init_ui()
        self.init_vars()

        if use_lqr_control:
            # liniearize around the up position
            self.setup_lqr(np.asarray([0., 0., np.pi, 0.]))

    def on_button_press(self, w, event):
        """When a button is pressed, the location gets stored and the canvas
        gets updated.
        """
        print(self.get_size())
        print("HERE!!!!!!!!!!!", event.x, event.y)
        print((event.x / self.get_size()[0] - .5) * self.SCALING)

        self.desired_x = -(event.x / self.get_size()[0] - .5) * self.SCALING / 2.5
        self.darea.queue_draw()

    def setup_lqr(self, y):
        self.use_lqr_control = True
        self.A, self.B = self.get_linearized_dynamics(y)
        Q = np.zeros((len(y), len(y)))
        Q[0, 0] = 1.; Q[1, 1] = 1.; Q[2, 2] = 10.; Q[3, 3] = 100.
        R = .001
        # compute gain matrix K
        self.K, _, _ = dlqr(self.A, self.B, Q, R)

    def init_ui(self):    

        self.darea = Gtk.DrawingArea()
        self.darea.set_size_request(1000,1000)
        self.darea.connect("draw", self.on_draw)
        self.darea.connect("button-press-event", self.on_button_press)
        self.darea.set_events(self.darea.get_events() |
                               Gdk.EventMask.BUTTON_MOTION_MASK |
                               Gdk.EventMask.BUTTON1_MOTION_MASK |
                               Gdk.EventMask.BUTTON2_MOTION_MASK |
                               Gdk.EventMask.BUTTON3_MOTION_MASK |
                               Gdk.EventMask.BUTTON_PRESS_MASK)
        self.add(self.darea)
        self.set_title("CartPole")
        self.resize(200, 200)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()

    def init_vars(self):   

        GLib.timeout_add(self.SPEED, self.on_timer)   
    
    def get_d(self, y, u, add_noise=True):

        M = self.M; m = self.m; L = self.L; g = self.g; d = self.d
        Sy = np.sin(y[2])
        Cy = np.cos(y[2])
        D = m*L*L*(M+m*(1-Cy**2))

        # some magic differential equations describing the evolution of the system
        d_p = y[1]
        d_v = (1./D)*(-m**2.*L**2.*g*Cy*Sy + m*L**2*(m*L*y[3]**2.*Sy - d*y[1])) + m*L*L*(1./D)*u
        d_a = y[3]
        r = np.random.normal() if add_noise else 0.
        d_av = (1./D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*y[3]**2.*Sy - d*y[1])) - m*L*Cy*(1/D)*u +.01*r

        return d_p, d_v, d_a, d_av

    def get_linearized_dynamics(self, y):
        eps = .01
        A = np.zeros((len(y), len(y)))
        for ii in range(len(y)):
            x_inc = np.zeros_like(y)
            x = copy.deepcopy(y)
            x[ii] += eps
            d_p, d_v, d_a, d_av = self.get_d(x, 0., add_noise=False)

            x_inc[0] = d_p + x[0]
            x_inc[1] = d_v + x[1]
            x_inc[2] = d_a + x[2]
            x_inc[3] = d_av + x[3]

            x_dec = np.zeros_like(y)
            x = copy.deepcopy(y)
            x[ii] -= eps
            d_p, d_v, d_a, d_av = self.get_d(x, 0., add_noise=False)
            x_dec[0] = d_p + x[0]
            x_dec[1] = d_v + x[1]
            x_dec[2] = d_a + x[2]
            x_dec[3] = d_av + x[3]
            A[:, ii] = (x_inc - x_dec) / (2 * eps)

        B = np.zeros((len(y), 1))

        u = eps
        x_inc = np.zeros_like(y)
        d_p, d_v, d_a, d_av = self.get_d(y, u, add_noise=False)
        x_inc[0] = d_p + y[0]
        x_inc[1] = d_v + y[1]
        x_inc[2] = d_a + y[2]
        x_inc[3] = d_av + y[3]

        u = -eps
        x_dec = np.zeros_like(y)
        d_p, d_v, d_a, d_av = self.get_d(y, u, add_noise=False)
        x_dec[0] = d_p + y[0]
        x_dec[1] = d_v + y[1]
        x_dec[2] = d_a + y[2]
        x_dec[3] = d_av + y[3]
        B[:, 0] = (x_inc - x_dec) / (2 * eps)

        return A, B

    def update_pendulum_dynamics(self, u):
        d_p, d_v, d_a, d_av = self.get_d(self.state, u)

        self.state[0] += d_p * self.dt
        self.state[1] += d_v * self.dt
        self.state[2] += d_a * self.dt
        self.state[3] += d_av * self.dt

    def get_control(self):
        if self.use_lqr_control:
            u = np.dot(-self.K, self.state - np.asarray([self.desired_x, 0, np.pi, 0.]))
            # TODO: 1.0 is not enough force? 1.5 decays to instability? What am
            # I doing wrong?
            return u * 1.0
        else:
            raise 0.

    def on_timer(self):
        u = self.get_control()
        self.update_pendulum_dynamics(u)
        self.darea.queue_draw()
        
        return True 

    def on_draw(self, wid, cr):

        w, h = self.get_size()

        cr.translate(w / 2, h / 2)
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