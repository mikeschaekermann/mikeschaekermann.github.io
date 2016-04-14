# Mike Schaekermann's Portfolio

Run locally using:

```
./tasks/serve
# then open the browser and go to http://127.0.0.1:4000/
```

Build notebooks using:

```
./tasks/notebooks/build_all
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