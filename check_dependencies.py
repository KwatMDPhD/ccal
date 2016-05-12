import imp
import pip
modules = ['numpy', 'pandas', 'matplotlib', 'seaborn']
for module in modules:
    try:
        imp.find_module(module)
        print('{} found.'.format(module))
    except ImportError:
        print('{} not found! Installing ......'.format(module))
        pip.main(['install', module])
