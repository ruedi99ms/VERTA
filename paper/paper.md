---
title: 'VERTA: a Python package for analyzing route choices in virtual environments'
tags:
    - Python
    - Virtual Reality
    - Mixed Reality
    - Augmented Reality
    - VR
    - XR
    - AR
    - Route Analysis
    - Behavioral Analysis
    - Evacuation
authors:
    -   name: Niklas Suhre
        orcid: 0009-0006-4168-1115
        affiliation: 1
    -   name: Meng Cai
        orcid: 0000-0002-8318-572X
        affiliation: 1
affiliations:
    -   name: Institute of Transport Planning and Traffic Engineering, Technical University of Darmstadt, Germany
        index: 1
bibliography: paper.bib
csl: apa_style.csl
---

# Summary

VERTA (Virtual Environment Route and Trajectory Analyzer) is a Python package that helps researchers understand how people choose routes in virtual environments. While originally developed for emergency evacuation research, VERTA can be applied to any study where route decisions in virtual environments need to be analyzed, such as wayfinding studies, navigation research, spatial cognition experiments, or urban planning simulations. When researchers study movement behavior using virtual reality, they collect data about where people move, but analyzing this movement data to identify decision-making patterns is complex. VERTA automates this analysis by detecting what route choices people make at intersections, identifying common routes taken, calculating timing and speed metrics, and predicting future route choices based on behavioral patterns. The software provides both a command-line interface for automated batch processing and a web-based graphical interface for interactive exploration. By standardizing how route choice data is analyzed, VERTA enables researchers to compare findings across different scenarios. This helps professionals in fields such as urban planning, emergency management, and human-computer interaction design more effective environments based on evidence from behavioral studies.

# Statement of need

As extreme events become more common due to climate change, the need for evacuations will grow as well [@Kuhl2014; @Thompson2017]. E.g., urban areas may need to be evacuated due to flooding, or buildings due to fires. Virtual Reality (VR) has become an important study tool in this context because it enables researchers to analyze people's behavior in controlled evacuation scenarios, including the crucial question of which routes people choose when there is limited or no guidance [@Hung2025].

Beyond evacuation research, understanding route choice behavior in virtual environments is relevant for a wide range of applications, including wayfinding studies, navigation research, spatial cognition experiments, urban planning simulations, and human-computer interaction research (e.g. @Ahmad2024100472[], @Li2019120[]). In all these domains, researchers face the same fundamental challenge: turning raw movement trajectories from VR experiments into meaningful insights about how and why people choose particular routes.

The Virtual Environment Route and Trajectory Analyzer (VERTA) addresses this gap by providing standardized, reproducible methods to analyze route choice behavior from VR experiments. VERTA detects where route decisions are made, summarizes which routes are preferred under different conditions, and enables systematic comparison across scenarios and experimental settings. This helps build an evidence base for designing more effective evacuation routes, wayfinding systems, and virtual environments. VERTA makes route choice analysis accessible to both technical and non-technical users through programmatic tools and an interactive graphical interface.

# State of the field

Current research on evacuation and pedestrian movement typically relies on a mix of custom scripts and general-purpose tools (e.g., R, Python, scikit-learn, microscopic crowd or traffic simulators) rather than a dedicated, standardized pipeline for route choice analysis in virtual environments. These tools are powerful, but they usually work either at an aggregate level (e.g., discrete choice models on already-encoded alternatives (e.g. @Hu2026105335[], @Lovreglio2022104452[])) or at the level of (simulated) flows (e.g. @Wang2024[], @Zhang2023106041[]). They do not directly help researchers analyze route choices at specific junctions or how those decisions unfold across multiple junctions in a standardized way. VERTA is intended to complement this existing ecosystem. Given user-defined junction locations and decision regions, it processes raw x–z VR trajectories and derives standardized decision outcomes (branch assignments), timing metrics, gaze- and physiology-based summaries, and junction-level patterns (including conditional probabilities and early intent predictions). This provides a shared, reproducible analysis layer that existing simulators and statistical models can build on.

# Software Design

VERTA uses a modular architecture that keeps the core analysis components (decision detection, clustering, metrics, prediction) separate from the user interface. We built the command-line interface first as researchers need reproducible, scriptable workflows for batch processing and comparing different evacuation scenarios. Later, we added a web-based GUI using Streamlit that uses the same underlying code, making the tool accessible to users who are not comfortable with command-line tools. The GUI includes advanced interactive features such as visual junction editing with click-to-add functionality, real-time parameter adjustment with live visual feedback, interactive conditional probability analysis showing route transition patterns between junctions, and interactive Plotly visualizations with zoom and pan capabilities. These features enhance exploratory analysis beyond what the CLI provides. The software offers multiple ways to detect decision points (pathlen, radial, hybrid) and different clustering methods (k-means, DBSCAN, automatic) as VR experiments produce varying outputs: people move at different speeds, pause at intersections, and make decisions in varied ways. Once decision points are identified, VERTA automatically detects branch directions by clustering the direction vectors extracted at these points. The k-means method requires specifying the number of expected branches, DBSCAN discovers variable numbers without prior specification, and the automatic method searches for the optimal number within a specified range. This automated branch detection eliminates manual route classification and enables consistent identification of common route choices. Figure 1 illustrates how VERTA identifies decision points along trajectories at user-defined junctions, showing the spatial distribution of route choices and the detected decision intercepts and route directions.

![Decision intercepts visualization showing multiple trajectories passing through a circular junction region with decision points and route directions marked along each path.](images/Decision_Intercepts.png)

This flexibility matters for research: keeping decision detection separate from branch assignment means we can compare results across studies, the modular design lets us add new features without breaking existing workflows, and having both CLI and GUI means technical researchers can automate their analyses while domain experts can explore data interactively with advanced analysis and visualization tools.

# Research Impact Statement

We have already used VERTA in the "RESCUE" (Routing Efficiency Study of 2D and 3D Cartography for Urban Evacuations in Emergencies) research project. It has given us valuable insights in the route choice behavior and evacuation efficiency of a simulated flooding event in Frankfurt am Main (Germany) [@Suhre2025] in addition to the evacuees' perspectives [@Cai2025].

# AI usage disclosure

This project utilized AI-assisted development tools for various aspects of the codebase:

- **Cursor** and **ChatGPT** were used for:
  - Code refactoring
  - GUI design
  - Test scaffolding
  - Documentation refactoring and elaboration
  - Paper refinement

# Acknowledgements

Funded by the European Union (ERC, scAInce, 101087218). Views and opinions expressed are
however those of the author(s) only and do not necessarily reflect those of the European Union or the
European Research Council Executive Agency. Neither the European Union nor the granting authority
can be held responsible for them.

Special thanks go to Isamu Lautenschläger for his insights into Machine Learning.

# References
    