import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def pendulum_rhs(t, y, m1, m2, L1, L2, b1, b2, g=9.81):
   
    theta1, dtheta1, theta2, dtheta2 = y

    delta = theta1 - theta2

    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    sin1  = np.sin(theta1)
    sin2  = np.sin(theta2)

    denom = (m1 + m2) * L1 - m2 * L1 * cos_d**2

    num1 = (
        m2 * L1 * dtheta1**2 * sin_d * cos_d          # центростремительный член
        + m2 * g * sin2 * cos_d                         # гравитация на m2
        + m2 * L2 * dtheta2**2 * sin_d                 # ускорение второго звена
        - (m1 + m2) * g * sin1                          # гравитация на m1
        - b1 * dtheta1 / L1                             # трение в шарнире 0
        + b2 * (dtheta2 - dtheta1) * cos_d / L1        # трение в шарнире 1
    )

    num2 = (
        - m2 * L2 * dtheta2**2 * sin_d * cos_d         # центростремительный член
        + (m1 + m2) * g * sin1 * cos_d                  # гравитация
        - (m1 + m2) * L1 * dtheta1**2 * sin_d           # ускорение первого звена
        - (m1 + m2) * g * sin2                           # гравитация на m2
        - b2 * (dtheta2 - dtheta1) / L2                 # трение в шарнире 1
    )

    ddtheta1 = num1 / denom
    ddtheta2 = num2 * L1 / (L2 * denom)

    return [dtheta1, ddtheta1, dtheta2, ddtheta2]


def simulate(m1, m2, L1, L2, b1, b2,
             theta1_0, theta2_0,
             dtheta1_0=0.0, dtheta2_0=0.0,
             t_end=10.0, dt=0.01, g=9.81):
    
    y0 = [theta1_0, dtheta1_0, theta2_0, dtheta2_0]

    t_span = (0.0, t_end)
    t_eval = np.arange(0.0, t_end, dt)

    sol = solve_ivp(
        fun=pendulum_rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(m1, m2, L1, L2, b1, b2, g),
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )

    theta1  = sol.y[0]
    dtheta1 = sol.y[1]
    theta2  = sol.y[2]
    dtheta2 = sol.y[3]

    x1 =  L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return {
        't':       sol.t,
        'theta1':  theta1,  'dtheta1': dtheta1,
        'theta2':  theta2,  'dtheta2': dtheta2,
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
    }



def plot_trajectory(result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(result['t'], np.degrees(result['theta1']), label='θ₁')
    axes[0].plot(result['t'], np.degrees(result['theta2']), label='θ₂')
    axes[0].set_xlabel('Время, с')
    axes[0].set_ylabel('Угол, градусы')
    axes[0].set_title('Углы во времени')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(result['x2'], result['y2'], linewidth=0.5, alpha=0.7)
    axes[1].plot(result['x2'][0], result['y2'][0],
                 'go', markersize=8, label='старт')
    axes[1].set_xlabel('x, м')
    axes[1].set_ylabel('y, м')
    axes[1].set_title('Траектория m2')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_aspect('equal')

    axes[2].plot(np.degrees(result['theta1']),
                 np.degrees(result['dtheta1']),
                 linewidth=0.5, alpha=0.7)
    axes[2].set_xlabel('θ₁, градусы')
    axes[2].set_ylabel('θ̇₁, град/с')
    axes[2].set_title('Фазовый портрет θ₁')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def animate_pendulum(result, L1, L2, speed=1.0):

    fig, ax = plt.subplots(figsize=(6, 6))
    L_total = L1 + L2
    ax.set_xlim(-L_total * 1.2, L_total * 1.2)
    ax.set_ylim(-L_total * 1.2, L_total * 1.2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Двузвенный маятник')

    line,   = ax.plot([], [], 'o-', lw=2, color='steelblue', markersize=8)
    trace,  = ax.plot([], [], '-', lw=0.5, alpha=0.4, color='red')
    ax.plot(0, 0, 'k^', markersize=10)

    trace_x, trace_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(i):
        xs = [0, result['x1'][i], result['x2'][i]]
        ys = [0, result['y1'][i], result['y2'][i]]
        line.set_data(xs, ys)

        trace_x.append(result['x2'][i])
        trace_y.append(result['y2'][i])
        trace.set_data(trace_x, trace_y)
        return line, trace

    step = max(1, int(1 / speed))
    frames = range(0, len(result['t']), step)

    ani = animation.FuncAnimation(
        fig, update, frames=frames,
        init_func=init, interval=20, blit=True
    )
    plt.show()
    return ani


if __name__ == '__main__':

    params = dict(
        m1=1.0, m2=1.0,
        L1=1.0, L2=0.5,
        b1=0.0, b2=0.0,  # вязкое трение
        theta1_0=1.0,
        theta2_0=0.5,
        t_end=15.0,
        dt=0.01
    )

    result = simulate(**params)

    print(f"Шагов в траектории: {len(result['t'])}")
    print(f"Время: {result['t'][0]:.2f} — {result['t'][-1]:.2f} с")

    #plot_trajectory(result)
    animate_pendulum(result, L1=params['L1'], L2=params['L2'])