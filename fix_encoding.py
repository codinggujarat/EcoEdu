import codecs

input_file = r'c:\GITHUB\EduEco\templates\login.html'

with codecs.open(input_file, 'r', 'utf-16le') as f:
    data = f.read()

with codecs.open(input_file, 'w', 'utf-8') as f:
    f.write(data)

print("Conversion complete.")
