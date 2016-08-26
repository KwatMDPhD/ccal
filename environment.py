import pip

print_log('Checking dependencies ...')
packages_installed = [pkg.key for pkg in pip.get_installed_distributions()]
packages_needed = []
for pkg in packages_needed:
    if pkg not in packages_installed:
        print_log('{} not found! Installing {} using pip ...'.format(pkg, pkg))
        pip.main(['install', pkg])
print_log('Using the following packages:')
for pkg in pip.get_installed_distributions():
    if pkg.key in packages_needed:
        print_log('\t{} (v{})'.format(pkg.key, pkg.version)

                  )