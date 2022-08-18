import setuptools

project_urls = {
  'AutoBrewML GitHub': 'https://github.com/microsoft/AutoBrewML'
}

# # read the contents of your README file
# from pathlib import Path
# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()
def readme():
    with open('README.rst') as f:
        return f.read()


setuptools.setup(
 name='AutoBrewML', 
 version='0.42',
 author="Sreeja Deb",
 author_email="srde@microsoft.com",
 long_description=readme(),
 description="With AutoBrewML Framework the time it takes to get production-ready ML models with great ease and efficiency highly accelerates.",
 packages=setuptools.find_packages(),
 project_urls = project_urls,
 install_requires=['numpy', 'pandas', 'sklearn','pandas_profiling','MarkupSafe==2.0.1','imblearn','scikit-learn==0.23.1','fairlearn','tpot','pyod'],
 classifiers=[
 "Programming Language :: Python :: 3",
 "License :: OSI Approved :: MIT License",
 "Operating System :: OS Independent",
 ],
)