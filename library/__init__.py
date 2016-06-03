# Check dependencies and install missing ones
import pip

packages_installed = [pkg.key for pkg in pip.get_installed_distributions()]
packages_needed = ['rpy2', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn']
for pkg in packages_needed:
    if pkg not in packages_installed:
        print('{} not found! Installing ......'.format(pkg))
        pip.main(['install', pkg])
print('Using the following packages:')
for pkg in pip.get_installed_distributions():
    if pkg.key in packages_needed:
        print('{} v{}'.format(pkg.key, pkg.version))
