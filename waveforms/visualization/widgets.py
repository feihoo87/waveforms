import matplotlib.pyplot as plt
import numpy as np


class DataPicker():

    def __init__(self, ax=None):
        self.points_and_text = []
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        self.ax = plt.gca()
        # 鼠标左键的button值为1
        if event.button == 1 and event.inaxes is not None:
            point, = self.ax.plot(event.xdata, event.ydata, 'ro')
            text = self.ax.text(event.xdata,
                                event.ydata,
                                f'({event.xdata:.2f}, {event.ydata:.2f})',
                                verticalalignment='center')
            self.points_and_text.append((point, text))
            self.ax.draw()

        elif event.button == 3 and event.inaxes is not None:
            for point, text in self.points_and_text:
                point_xdata, point_ydata = point.get_data()
                point_xdata, point_ydata = point_xdata[0], point_ydata[0]
                point_xdisplay, point_ydisplay = self.ax.transData.transform_point(
                    [point_xdata, point_ydata])
                event_xdisplay, event_ydisplay = event.x, event.y

                distance = np.sqrt((point_xdisplay - event_xdisplay)**2 +
                                   (point_ydisplay - event_ydisplay)**2)
                if distance < 10:
                    point.remove()
                    text.remove()
                    self.points_and_text.remove((point, text))
                    self.ax.draw()
                    break

    def get_xy(self):
        x, y = [], []
        for point, text in self.points_and_text:
            point_xdata, point_ydata = point.get_data()
            x.extend(point_xdata)
            y.extend(point_ydata)

        index = np.argsort(x)
        x = np.asarray(x)[index]
        y = np.asarray(y)[index]
        return x, y
