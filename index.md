---
layout: home
title: "My Data Science Blog"
---

# Welcome to My Blog

Here are my latest posts:

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) — *{{ post.date | date_to_string }}*
{% endfor %}
