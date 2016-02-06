---
layout: page
title: Articles
tags: [articles, blog, posts]
modified: 2016-02-06T20:53:07.573882-04:00
share: false
---

This is a collection of articles, blog posts, drafts and notes of mine that I thought might be worth sharing publicly:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> (from {{ post.date | date: "%B %d, %Y" }})
    </li>
  {% endfor %}
</ul>