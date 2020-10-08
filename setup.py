import setuptools

setuptools.setup(
    name="CycleFlow",
    version="1.0",
    author="Adrien Jolly and Nils B. Becker",
    author_email="a.jolly@dkfz.de",
    description="Quantification of Cell Cycle length and cycling heterogeneity",
    url="https://github.com/AdrienJolly/CycleFlow",
    install_requires=['numpy', 'numba','scipy','pandas']
)
