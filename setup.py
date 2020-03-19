from setuptools import setup, find_packages

setup(
	name='cnns4qspr',
	description='A package for creating rich, equivariant, structural and chemical features from protein structure data.',
    	url='https://https://github.com/CNNs4QSPR/cnns4qspr',
    	author=[
		'Orion Dollar',
		'David Juergens',
		'Nisarg Joshi',
		'Saransh Jain'
		],
    	author_email=[
		'orion.dollar@gmail.com',
		'davidcj@uw.edu',
		'nisargj@uw.edu',
		'saranshj$uw.edu',
		],
	license='MIT License',
	packages=find_packages(),
    	install_requires=[
			'numpy',
			'scipy',
			'torch',
			'pandas',
			'plotly.express',
			'biopandas',
			'se3cnn',
			'lie_learn',
        		],
	dependency_links=[
			'https://github.com/AMLab-Amsterdam/lie_learn',
			'https://github.com/blondegeek/se3cnn',
			],
    	classifiers=[
        	'Intended Audience :: Science/Research',
       		'License :: OSI Approved :: MIT License',
        	'Operating System :: POSIX :: Linux',
        	'Programming Language :: Python :: 3.7',
    		],
)
