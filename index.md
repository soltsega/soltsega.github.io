---
layout: home
title: "My Data Science Blog"
---

<!-- MathJax script for rendering LaTeX equations -->
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Welcome to My Blog

Here are my latest posts:

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) — *{{ post.date | date_to_string }}*
{% endfor %}
