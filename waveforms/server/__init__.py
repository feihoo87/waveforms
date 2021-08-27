import click

class App():
    def __init__(self, **kwds):
        self.kwds = kwds

    def run(self):
        click.echo("Running waveforms server")
        click.echo("Server is running")
        for key, values in self.kwds.items():
            click.echo(f"    {key} : ", nl=False)
            click.secho(f"{values}", fg='blue')
        click.pause()
        MARKER = '# Everything below is ignored\n'
        message = click.edit('\n\n' + MARKER)
        if message is not None:
            click.echo(message.split(MARKER, 1)[0].rstrip('\n'))


def create_app(**kwds):
    return App(**kwds)
