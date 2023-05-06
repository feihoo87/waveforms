import pickle
import pathlib
import click
import matplotlib.pyplot as plt
import numpy as np

from .qdat import draw as draw_qdat

default_draw_methods = {
    '.qdat': draw_qdat,
}


def load_data(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def draw(data):
    script = data['meta']['plot_script']
    global_namespace = {'plt': plt, 'np': np, 'result': data}
    exec(script, global_namespace)


def draw_error(data, text="No validate plot script found"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.set_axis_off()
    return fig


@click.command()
@click.argument('fname', default='')
def plot(fname):
    """Plot the data in the file."""
    try:
        fname = pathlib.Path(fname)
        data = load_data(fname)
        try:
            draw(data)
        except:
            default_draw_methods.get(fname.suffix, draw_error)(data)
    except FileNotFoundError:
        draw_error(None, text=f"File {fname} not found.")
    except pickle.UnpicklingError:
        draw_error(None, text=f"File {fname} is not a pickle file.")

    plt.show()


if __name__ == '__main__':
    plot()
