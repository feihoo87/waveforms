import click
from .visualization.__main__ import plot


@click.group()
def main():
    pass


@main.command()
def hello():
    """Print hello world."""
    click.echo('hello, world')


main.add_command(plot)

if __name__ == '__main__':
    main()
