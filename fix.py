import glob
for f in glob.glob('templates/admin_*.html'):
    with open(f, 'r', encoding='utf-8') as file:
        c = file.read()
    nc = c.replace('{% extends"adminbase.html" %}', '{% extends "adminbase.html" %}')
    if nc != c:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(nc)
        print(f'Fixed {f}')
