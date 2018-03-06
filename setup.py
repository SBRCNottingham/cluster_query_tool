from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


long_description = '''
'''

setup(
    name="cqt",
    version="0.0.1",
    description="Cluster query tool for complex networks",
    long_description=long_description,
    zip_safe=False,
    author="James Gilbert",
    install_requires=requirements,
    author_email="jamie.gilbert@azimov.co.uk",
    license="GPL",
    entry_points={},
    url="https://github.com/azimov/cluster_query_tool",
    include_package_data=True,
    packages=find_packages(),
    project_urls=dict(),
)
