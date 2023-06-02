from waveforms.math.fit import install_classify_method

from .state_classify import circular_region_classify, nearest_classify

install_classify_method('multilevel', nearest_classify)
