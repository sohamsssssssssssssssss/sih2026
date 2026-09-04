Create a complete **clickable Figma UI prototype** for a frontend project called **SatQuery AI**.

SatQuery AI is a natural-language assistant for analysing satellite imagery.

This is a **frontend-only demonstration**. There is no real backend, AI model, image processing, API, database, or actual satellite analysis. All files, results, workflows, and imagery shown in the prototype are demonstration data.

The prototype should feel like a **real engineering project / student-built MVP** that could be demonstrated at an ISRO/SAC presentation.

## DESIGN STYLE

Create a UI that is:

* Clean
* Practical
* Technical
* Trustworthy
* Modern but not futuristic
* Easy for a non-technical person to understand
* Suitable for a technical presentation

Use a **dark navy interface** with restrained blue/cyan accents.

The design should resemble a real application used for satellite-image analysis rather than an AI startup landing page.

### Avoid

Do NOT use:

* Excessive gradients
* Glassmorphism everywhere
* Neon glowing cards
* 3D illustrations
* Giant AI brain graphics
* Holograms
* Floating futuristic objects
* Excessive rounded pills
* Huge hero typography
* Overly decorative animations
* Generic “AI SaaS” layouts
* Too many icons
* Fake statistics
* Fake accuracy percentages
* Fake processing speeds

Keep the UI slightly dense and functional, like a real technical dashboard.

Use:

* Flat/dark surfaces
* Thin borders
* Small corner radius
* Simple icons
* Clear spacing
* Normal button sizes
* Compact metadata
* Practical labels
* Small satellite/map visualisations

Typography should be clean and readable. Use different font sizes and weights for hierarchy, but avoid oversized headings.

---

# GLOBAL HEADER

Every application screen should have a simple header.

Left:

**SatQuery AI**

Small text:

**Satellite Query Assistant**

Right:

**Demo Mode**

**● System Ready**

Add a small navigation/progress indicator where useful.

---

# SCREEN 1 — HOME

Create a simple but polished landing screen.

Left side:

**SatQuery AI**

**Ask questions. Understand Earth from space.**

Description:

**A natural-language interface for exploring satellite imagery through automated task selection and visual evidence.**

Primary button:

**Start Analysis →**

Small label:

**Frontend Demonstration**

Right side:

Create a realistic-looking **mock satellite image panel**.

The image should contain a mixture of:

* vegetation
* roads
* water
* built-up areas

Add very subtle grid lines and coordinate markings.

The satellite image should look like an analysis preview, not a decorative background.

Below the main section, show three compact cards:

### Single Image

Analyse one satellite image.

### Bi-temporal Change

Compare imagery from two dates.

### Optical + SAR

Combine optical and SAR imagery.

Click:

**Start Analysis → Screen 2**

---

# SCREEN 2 — UPLOAD IMAGERY

This screen should look like a real application workspace.

Page title:

**New Analysis**

Small progress indicator:

**01 Upload → 02 Validate → 03 Query → 04 Results**

Section:

**Select Analysis Mode**

Create three selectable options:

### Single Image Analysis

Analyse one satellite image and describe the land-cover and major objects visible.

### Bi-temporal Change Analysis

Compare imagery from two dates to identify and describe changes.

Badge:

**Recommended Demo**

### Optical + SAR Fusion

Combine optical and SAR imagery to identify complementary land-cover information.

---

## UPLOAD SECTION

For the selected Bi-temporal mode, show two practical upload cards.

### BEFORE IMAGE

**Upload Before Image**

**Drag and drop your image here**

**or**

**Browse Files**

Supported:

**GeoTIFF / TIFF**

**PNG / JPEG for approved benchmark inputs**

### AFTER IMAGE

**Upload After Image**

**Drag and drop your image here**

**or**

**Browse Files**

Supported:

**GeoTIFF / TIFF**

**PNG / JPEG for approved benchmark inputs**

Use a simple upload/cloud icon.

Make the whole upload area clickable.

---

## DEMO UPLOAD INTERACTION

Because this is a Figma prototype, clicking **Browse Files** should open a small modal.

Modal title:

**Select Demo Image**

Text:

**Choose an example image for this demonstration.**

Show three image options with thumbnails:

**Scene 01**
`scene_01.tif`

**Scene 02**
`scene_02.tif`

**Urban Scene**
`urban_scene.tif`

Buttons:

**Cancel**

**Use Selected Image**

After selecting one, show the uploaded state:

✓ **Uploaded**

Thumbnail

`scene_before.tif`

**GeoTIFF**

**Replace | Remove**

Repeat for the After image.

When both images are selected, display:

**✓ 2 images ready for analysis**

Enable:

**Continue →**

---

# SCREEN 3 — INPUT VALIDATION

Title:

**Check Inputs**

Subtitle:

**SatQuery AI checks the selected imagery before starting the analysis.**

Create a compact technical checklist:

✓ **File format** — Valid

✓ **Image count** — Valid

✓ **Sensor modality** — Compatible

✓ **Acquisition dates** — Available

✓ **Scene compatibility** — Compatible

✓ **Co-registration / alignment** — Verified where relevant

Use small green check icons.

Do not use huge glowing checkmarks.

Right side:

### Input Summary

**Before**

`scene_before.tif`

**After**

`scene_after.tif`

**Mode**

Bi-temporal Change Analysis

Bottom information box:

**Validation complete**

**The selected inputs are ready for query-based analysis.**

Button:

**Proceed to Query →**

---

# SCREEN 4 — ASK A QUESTION

Title:

**Ask a Question**

Subtitle:

**Ask about your imagery in natural language. SatQuery AI will automatically determine the appropriate analysis workflow.**

Create a large simple text area.

Placeholder:

**What would you like to know about these images?**

Below:

**Try an example**

Create three clickable prompt cards/chips:

**Describe the land-cover and major objects visible in this image.**

**What changed between these two dates, and where did the change occur?**

**Use the optical and SAR images together to identify built-up and water-covered regions.**

Clicking an example should populate the query field.

Helper text:

**You don't need to select a model or tool manually.**

Button:

**Run Analysis →**

---

# SCREEN 5 — AGENTIC ANALYSIS

Make this screen look like a **technical workflow monitor**, not a futuristic AI animation.

Title:

**Analysis Workflow**

Subtitle:

**SatQuery AI is interpreting your request and selecting the appropriate specialist workflow.**

Show a vertical process:

### 01

**Interpreting query**

Understanding the user's intent.

✓ Completed

### 02

**Validating inputs**

Checking imagery type, dates, and compatibility.

✓ Completed

### 03

**Selecting task**

Identifying the required analysis task.

✓ Completed

### 04

**Selecting specialist tools/models**

Choosing suitable specialist workflows.

✓ Completed

### 05

**Generating visual evidence**

Preparing comparison views and change masks.

✓ Completed

### 06

**Preparing answer**

Converting the analysis into a plain-language response.

✓ Completed

On the right, show:

### Selected Workflow

**Change-detection model + Change-description model**

Below:

**Workflow selected automatically**

**Processing completed**

Button:

**View Results →**

Make the automatic selection visually obvious.

---

# SCREEN 6 — RESULTS

Create a realistic technical results dashboard.

Title:

**Analysis Results**

Small metadata:

**Bi-temporal Change Analysis**

**Demonstration output**

---

## USER QUERY

“What changed between these two dates, and where did the change occur?”

---

## AI ANSWER

**Built-up area increased in the southern and eastern portions of the scene.**

Beside the answer:

**Confidence: High**

Do not use numeric confidence.

---

## INPUT DETAILS

Create a compact section:

**Before:** `scene_before.tif`

**After:** `scene_after.tif`

**Analysis Mode:** Bi-temporal Change Analysis

**Status:** Processing completed

---

## SELECTED WORKFLOW

**Change-detection model + Change-description model**

Small description:

**Automatically selected based on the user's question and available imagery.**

---

# VISUAL EVIDENCE

Make this the main visual element.

Create three equal image panels:

### BEFORE

Mock satellite image.

### AFTER

Mock satellite image showing increased built-up areas.

### DETECTED CHANGE

Comparison image with highlighted change regions.

The change should appear primarily in:

* southern portion
* eastern portion

Use:

**Red = Built-up expansion**

**Yellow = Change area**

Add small labels such as:

`BEFORE`

`AFTER`

`CHANGE MASK`

The images are **demonstration visuals**, not real processed satellite results.

---

# EXECUTION SUMMARY

Create a collapsible technical section:

**Execution Summary +**

When opened:

`Query interpreted`

`Inputs validated`

`Task selected`

`Specialist workflow selected`

`Visual evidence generated`

`Answer prepared`

Make it look like a simple application log.

---

# RESULT ACTIONS

Buttons:

**Download Report**

**Ask Another Question**

**New Analysis**

Interactions:

Download Report → Screen 7

Ask Another Question → Screen 4

New Analysis → Screen 2

---

# SCREEN 7 — REPORT

Create a simple completion page.

Success icon.

Title:

**Analysis Report Ready**

Description:

**The demonstration report contains the query, inputs, selected workflow, result, and visual evidence.**

Report preview:

### SatQuery AI Analysis Report

**Query**

What changed between these two dates, and where did the change occur?

**Inputs**

Before + After satellite imagery

**Workflow**

Change-detection model + Change-description model

**Result**

Built-up area increased in the southern and eastern portions of the scene.

**Evidence**

Before / After / Detected Change

Label:

**Demonstration output**

Buttons:

**Download Analysis Report**

**New Analysis**

When Download is clicked, show a small normal notification:

**Demo report download triggered**

---

# OTHER DEMO FLOWS

The prototype must also support the following modes.

## SINGLE IMAGE FLOW

Query:

**Describe the land-cover and major objects visible in this image.**

Answer:

**The image contains agricultural land, a water body, scattered built-up areas, and a road network.**

Workflow:

**Single-image VQA + Scene Description**

Visual evidence:

One satellite image with simple overlays for:

* Water
* Vegetation
* Buildings
* Roads

---

## BI-TEMPORAL FLOW

This should be the primary demonstration.

Query:

**What changed between these two dates, and where did the change occur?**

Answer:

**Built-up area increased in the southern and eastern portions of the scene.**

Workflow:

**Change-detection model + Change-description model**

Visual evidence:

**Before | After | Detected Change**

---

## OPTICAL + SAR FLOW

Query:

**Use the optical and SAR images together to identify built-up and water-covered regions.**

Answer:

**The fused analysis identifies water-covered regions in the western zone and dense built-up structures in the central-east zone.**

Workflow:

**Cross-modal Optical–SAR Analysis**

Visual evidence:

**Optical | SAR | Fused Result**

Use simple overlays for:

**Water**

**Built-up**

---

# HUMAN-CODED / REALISTIC UI DETAILS

The prototype should have the visual character of a **real working MVP**.

Include small details such as:

* File names
* Upload states
* Empty states
* Validation states
* Status labels
* Simple technical metadata
* Small legends
* Clear disabled/enabled buttons
* Practical error/success messaging
* Compact information panels
* Consistent spacing
* Simple icons

Do not make every section perfectly symmetrical.

Do not fill empty space with unnecessary decorative graphics.

Do not add fake metrics.

Do not add fake accuracy.

Do not add fake processing time.

Do not add numeric confidence scores.

Use only:

**Confidence: High / Medium / Low**

**Processing completed**

**Demonstration output**

The final prototype should feel like:

**“A group of engineering students built a genuinely usable satellite-analysis application.”**

It should NOT feel like:

**“An AI generated a futuristic dashboard concept.”**

The overall user story must be immediately clear:

**Upload imagery → Ask a question → SatQuery AI interprets it → Automatically selects specialist workflows → Generates visual evidence → Gives a plain-language answer → Creates a report.**
