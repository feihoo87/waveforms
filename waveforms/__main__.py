import click

from .sys.net.cli import dht


@click.group()
def main():
    pass


@main.command()
def hello():
    """Print hello world."""
    click.echo('hello, world')


main.add_command(dht)

if __name__ == '__main__':
    main()
