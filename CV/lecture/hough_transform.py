import numpy as np
import plotly.graph_objects as go


points = [(10, 10), (20, 20), (30, 30)]
thetas = np.linspace(0, np.pi, 100)


def hough_space():
    fig = go.Figure()
    for (x, y) in points:
        rhos = x * np.cos(thetas) + y * np.sin(thetas)
        fig.add_trace(go.Scatter(x=thetas, y=rhos, mode='lines', name=f'Point ({x}, {y})'))
    fig.add_trace(go.Scatter(x=[3 / 4 * np.pi], y=[0], mode='markers', marker=dict(color='red', size=10), name='Intersection Point'))
    fig.update_layout(
        title='Hough Transform Curves for Points',
        xaxis_title='Theta (radians)',
        yaxis_title='Rho',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    )
    fig.show()

def original_space():
    x, y = zip(*points)
    xx = np.linspace(-3 / np.sqrt(2), 3, 100)
    yy = np.sqrt(9 - xx ** 2)
    fig = go.Figure(data=[
        go.Scatter(x=x, y=y, mode='markers', name='Original Points'),
        go.Scatter(x=xx, y=yy, mode='lines', name='Angle'),
        go.Scatter(x=np.linspace(-10, 40, 100), y=np.linspace(-10, 40, 100), mode='lines', name='Line'),
        go.Scatter(x=np.linspace(-10, 10, 40), y=np.linspace(10, -10, 40), mode='lines', name='Orthogonal Line'),
    ])
    fig.update_layout(
        title='Original Space with Points and Lines',
        xaxis_title='X',
        yaxis_title='Y',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    )
    fig.show()


original_space()