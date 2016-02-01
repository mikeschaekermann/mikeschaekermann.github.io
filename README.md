# Mike Schaekermann's Portfolio

Run locally using:

```
bundle exec jekyll serve
# then open the browser and go to http://localhost:4000/
```

Deploy using:

```
# build site
bundle exec jekyll build --config _config.yml,_production.yml
# copy site to web server
scp -r _site/* mschaeke@linux.cs.uwaterloo.ca:/u3/mschaekermann/public_html
```

Redirect from home to specific sub page:

```
# .htaccess file in public_html
RewriteEngine On
RewriteRule ^$ /~mschaeke/projects/ [R]
```