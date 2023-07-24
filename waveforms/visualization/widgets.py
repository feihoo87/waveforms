import matplotlib.pyplot as plt
import numpy as np


class DataPicker():

    def __init__(self, ax=None):
        self.points_and_text = {}
        self.line = None
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.mode = 'pick'

    def on_key_press(self, event):
        if event.key == 'p':
            if self.mode != 'pick':
                self.mode = 'pick'
            else:
                self.mode = 'default'

    def on_click(self, event):
        if self.mode != 'pick':
            return
        # 鼠标左键的button值为1
        if event.button == 1 and event.inaxes is not None:
            point = (event.xdata, event.ydata)
            text = self.ax.text(point[0],
                                point[1],
                                f'({point[0]:.2f}, {point[1]:.2f})',
                                verticalalignment='center')
            self.points_and_text[point] = text
            x, y = self.get_xy()
            if self.line is None:
                self.line, = self.ax.plot(x, y, 'ro')
            else:
                self.line.set_data(x, y)
            self.ax.draw()

        elif event.button == 3 and event.inaxes is not None:
            for point, text in list(self.points_and_text.items()):
                point_xdisplay, point_ydisplay = self.ax.transData.transform_point(
                    point)

                distance = np.sqrt((point_xdisplay - event.x)**2 +
                                   (point_ydisplay - event.y)**2)
                if distance < 10:
                    text.remove()
                    self.points_and_text.pop(point)
                    if self.points_and_text:
                        x, y = self.get_xy()
                        self.line.set_data(x, y)
                    else:
                        self.line.remove()
                        self.line = None
                    self.ax.draw()
                    break

    def get_xy(self):
        if self.points_and_text:
            data = np.asarray(list(self.points_and_text.keys()))
            x, y = data[:, 0], data[:, 1]

            index = np.argsort(x)
            x = np.asarray(x)[index]
            y = np.asarray(y)[index]
            return x, y
        else:
            return np.array([]), np.array([])
