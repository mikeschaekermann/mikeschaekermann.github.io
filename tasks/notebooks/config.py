# modification of config created here: https://gist.github.com/cscorley/9144544
try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        f = arg.split('.ipynb')[0]
        break

c = get_config()
c.NbConvertApp.export_format = 'markdown'
c.MarkdownExporter.template_path = [os.path.join(root_directory, '_layouts')] # point this to your jekyll template file
c.MarkdownExporter.template_file = 'notebook'
#c.Application.verbose_crash=True

# modify this function to point your images to a custom path
# by default this saves all images to a directory 'images' in the root of the blog directory
def path2support(path):
    """Turn a file path into a URL"""
    return '{{ site.url }}/images/' + os.path.basename(path)

c.MarkdownExporter.filters = {'path2support': path2support}

if f:
    c.NbConvertApp.output_base = f.lower().replace(' ', '-')
    c.FilesWriter.build_directory = root_directory # point this to your build directory