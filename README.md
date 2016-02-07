# Mike Schaekermann's Portfolio

Run locally using:

```
bundle exec jekyll serve
# then open the browser and go to http://127.0.0.1:4000/
```

Deploy using:

```
./tasks/deploy
```

Redirect from home to specific sub page:

```
# .htaccess file in public_html
RewriteEngine On
RewriteRule ^$ /~mschaeke/projects/ [R]
```