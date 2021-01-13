import setuptools

setuptools.setup(
    name="CycleFlow",
    version="1.1",
    author="Adrien Jolly and Nils B. Becker",
    author_email="a.jolly@dkfz.de",
    description="Quantification of Cell Cycle length and cycling heterogeneity",
    url="https://github.com/AdrienJolly/CycleFlow",
    install_requires=['numpy', 'numba','scipy','pandas'],
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                ]
)
