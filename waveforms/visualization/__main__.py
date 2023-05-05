import pickle
import pathlib
import click
import matplotlib.pyplot as plt
import numpy as np

from .qdat import draw as draw_qdat

default_draw_methods = {
    '.qdat': draw_qdat,
}


def draw(data):
    script = data['meta']['plot_script']
    global_namespace = {'plt': plt, 'np': np, 'result': data}
    exec(script, global_namespace)


def draw_error(data, text="No plot script found"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.set_axis_off()
    return fig


@click.command()
@click.argument('fname')
def main(fname):
    fname = pathlib.Path(fname)
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
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
    main()
