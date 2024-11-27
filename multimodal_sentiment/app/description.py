"""
File: description.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Project description for the Gradio app.
License: MIT License
"""

# Importing necessary components for the Gradio app
from app.config import config_data

DESCRIPTION_STATIC = f"""\
# 面部表情分析
<div class="app-flex-container">
    <img src="https://img.shields.io/badge/version-v{config_data.APP_VERSION}-rc0" alt="Version">
    <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FElenaRyumina%2FFacial_Expression_Recognition"><img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FElenaRyumina%2FFacial_Expression_Recognition&countColor=%23263759&style=flat" /></a>
    <a href="https://paperswithcode.com/paper/in-search-of-a-robust-facial-expressions"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-search-of-a-robust-facial-expressions/facial-expression-recognition-on-affectnet" /></a>
    </div>
"""

DESCRIPTION_DYNAMIC = f"""\
# 面部表情分析
<div class="app-flex-container">
    <img src="https://img.shields.io/badge/version-v{config_data.APP_VERSION}-rc0" alt="Version">
    <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FElenaRyumina%2FFacial_Expression_Recognition"><img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FElenaRyumina%2FFacial_Expression_Recognition&countColor=%23263759&style=flat" /></a>
    <a href="https://paperswithcode.com/paper/in-search-of-a-robust-facial-expressions"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-search-of-a-robust-facial-expressions/facial-expression-recognition-on-affectnet" /></a>
    </div>
"""
