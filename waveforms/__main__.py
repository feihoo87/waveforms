import click


@click.group()
def main():
    pass


@main.command()
@click.option('--sample-rate', '-S', default=44100, help='Sample rate in Hz')
@click.option('--start', '-a', default=0, help='Start time in seconds')
@click.option('--duration', '-l', default=-1, help='Duration in seconds')
@click.option('--stop', '-b', default=1, help='Stop time in seconds')
@click.option('--amplitude', '-A', default=1, help='Amplitude')
@click.argument('waveform', type=str)
@click.argument('output', type=click.Path(exists=False))
def sample(sample_rate, start, duration, stop, amplitude, waveform, output):
    """Generate a waveform sample."""
    import numpy as np

    from waveforms import wave_eval

    wav = wave_eval(waveform)
    wav.start = start
    if duration > 0 and stop == 1:
        stop = start + duration
    wav.stop = stop
    wav.sample_rate = sample_rate
    points = wav.sample()
    points = points * amplitude
    np.save(output, points)


if __name__ == '__main__':
    main()
