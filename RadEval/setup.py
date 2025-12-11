from setuptools import setup, find_packages

setup(
    name='RadEval',
    version='0.0.1',
    author='Jean-Benoit Delbrouck, Justin Xu, Xi Zhang',
    maintainer='Xi Zhang',
    url='https://github.com/jbdel/RadEval',
    project_urls={
        'Bug Reports': 'https://github.com/jbdel/RadEval/issues',
        'Source': 'https://github.com/jbdel/RadEval',
        'Documentation': 'https://github.com/jbdel/RadEval/blob/main/README.md',
    },
    license='MIT',
    description='All-in-one metrics for evaluating AI-generated radiology text',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords=[
        'radiology',
        'evaluation',
        'natural language processing',
        'radiology report',
        'medical NLP',
        'clinical text generation',
        'LLM',
        'bioNLP',
        'chexbert',
        'radgraph',
        'medical AI'
    ],
    python_requires='>=3.9,<=3.12.12',
    install_requires=[
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "RadEval.factual.SRRBert": ["*.json"],
    },
    zip_safe=False,
    )
