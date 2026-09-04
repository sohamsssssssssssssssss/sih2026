## REDESIGN SATQUERY AI — MAKE IT LOOK HUMAN-MADE, NOT AI-GENERATED

Redesign the existing SatQuery AI prototype so it looks like a **real student-built technical project / working MVP**, not an AI-generated concept UI.

The interface should feel like something a small engineering team could realistically build with **HTML, CSS and JavaScript** and present at a technical competition or ISRO/SAC-related demonstration.

### MOST IMPORTANT DESIGN RULE

**Do NOT make it look like a generic AI SaaS landing page.**

Avoid the typical AI-generated visual patterns:

* No huge glowing gradients
* No excessive glassmorphism
* No floating 3D objects
* No excessive neon
* No giant decorative AI graphics
* No unnecessary animated blobs
* No futuristic holograms
* No excessive rounded “pill” components
* No overly perfect marketing-style sections
* No stock-photo-looking satellite backgrounds
* No excessive shadows
* No giant typography taking up half the screen
* No “AI brain” illustrations
* No unnecessary icons everywhere

Instead, make it feel **functional, technical and believable**.

---

# VISUAL PERSONALITY

Think:

**“A real satellite-analysis dashboard built by engineering students.”**

Not:

**“A futuristic AI startup website.”**

Use a restrained dark interface.

Background:

* Deep navy
* Almost charcoal in some areas
* Mostly flat surfaces

Cards:

* Slightly lighter navy
* Thin borders
* Small corner radius
* Very subtle shadow or no shadow

Accent:

* One primary blue/cyan accent
* Green only for successful validation
* Yellow/red only where they have functional meaning

Typography:

* Clean sans-serif
* Normal font weights
* Avoid oversized headings
* Strong hierarchy through spacing and font size rather than decoration

Keep the interface compact and information-dense, similar to a real engineering dashboard.

---

# HEADER

Use a simple top navigation bar.

Left:

**SatQuery AI**

Small text beside it:

**Satellite Query Assistant**

Right:

**Demo Mode**

and a simple status indicator:

**● System Ready**

Do not make the header oversized.

---

# HOME SCREEN

Make the home page much simpler.

Instead of a huge hero section, use a two-column layout.

Left:

**SatQuery AI**

**Ask questions. Understand Earth from space.**

Short description:

**A natural-language interface for exploring satellite imagery through automated task selection and visual evidence.**

Button:

**Start Analysis →**

Right:

A simple **satellite-image/map preview panel**.

Inside it, show a believable mock satellite scene with:

* roads
* vegetation
* water
* buildings
* small coordinate markings
* subtle grid

Make it look like an analysis preview rather than a decorative hero image.

Below the hero, show three compact modules:

**Single Image**

**Bi-temporal Change**

**Optical + SAR**

Keep them practical and small.

---

# UPLOAD SCREEN

Make this screen feel like an actual application.

Header:

**New Analysis**

Small progress indicator:

**01 Upload → 02 Validate → 03 Query → 04 Results**

Do not use huge decorative graphics.

### Analysis Mode

Use three compact selectable tabs/cards:

**Single Image**

**Bi-temporal Change**

**Optical + SAR**

The selected mode should have a simple border/accent rather than a glowing effect.

---

## Upload Area

For Bi-temporal Change Analysis, create:

### Before Image

Large but practical upload box.

Show:

**Upload Before Image**

**Drag and drop or browse**

`GeoTIFF / TIFF`

Button:

**Browse Files**

### After Image

Same structure.

Make the upload boxes look like something a developer would realistically implement.

Use a simple upload icon.

When an image is uploaded, change the card to:

**✓ Uploaded**

`scene_before.tif`

`GeoTIFF`

Show a small thumbnail.

Actions:

**Replace** | **Remove**

Do NOT make the upload area look like a futuristic AI component.

---

# DEMO FILE PICKER

Since this is a Figma prototype, clicking **Browse Files** should open a simple modal.

Title:

**Select Demo Image**

Text:

**Choose an example image for this prototype.**

Show three small thumbnail rows:

**Scene 01**
`scene_01.tif`

**Scene 02**
`scene_02.tif`

**Urban Scene**
`urban_scene.tif`

Button:

**Use Selected Image**

Secondary:

**Cancel**

Make this modal look like a normal application dialog.

---

# VALIDATION SCREEN

Title:

**Check Inputs**

Show a straightforward technical checklist.

✓ File format
✓ Image count
✓ Sensor modality
✓ Acquisition dates
✓ Scene compatibility
✓ Co-registration / alignment

Each item should have:

**Status**

**Valid**

Avoid large green glowing checkmarks.

Use small check icons and compact rows.

On the right, show:

**Input Summary**

Before:

`scene_before.tif`

After:

`scene_after.tif`

Mode:

**Bi-temporal Change Analysis**

Bottom button:

**Proceed to Query →**

---

# QUERY SCREEN

Title:

**Ask a Question**

Keep this screen extremely simple.

Large textarea:

**What would you like to know about these images?**

Below it:

**Try an example**

Then three clickable examples:

> Describe the land-cover and major objects visible in this image.

> What changed between these two dates, and where did the change occur?

> Use the optical and SAR images together to identify built-up and water-covered regions.

When clicked, populate the query field.

Below:

**SatQuery AI will automatically select the appropriate analysis workflow.**

Button:

**Run Analysis →**

---

# AGENTIC ANALYSIS SCREEN

Do NOT make this look like a futuristic AI animation.

Instead, make it resemble a **developer/engineering workflow monitor**.

Title:

**Analysis Workflow**

Small text:

**SatQuery AI is determining the appropriate workflow for your query.**

Show a vertical list:

**01 — Interpreting query**
Completed

**02 — Validating inputs**
Completed

**03 — Selecting task**
Completed

**04 — Selecting specialist tools/models**
Completed

**05 — Generating visual evidence**
Completed

**06 — Preparing answer**
Completed

Use simple progress indicators.

On the right, show:

### Selected Workflow

**Change-detection model + Change-description model**

Below:

**Why this workflow?**

**The system identified a bi-temporal change-analysis task from the user's question and selected the corresponding specialist workflow.**

Bottom:

**Processing completed**

Button:

**View Results →**

This explanation is important because it makes the **agentic orchestration** obvious.

---

# RESULTS SCREEN

Make this look like an actual technical analysis result page.

Header:

**Analysis Results**

Top small metadata:

**Bi-temporal Change Analysis**
**Demonstration output**

---

## QUERY

Use a simple bordered section:

**User Query**

“What changed between these two dates, and where did the change occur?”

---

## ANSWER

Make the answer the main focus but do NOT use a giant AI chat bubble.

**Built-up area increased in the southern and eastern portions of the scene.**

Beside it:

**Confidence: High**

---

# VISUAL EVIDENCE

This should be the strongest visual section.

Title:

**Visual Evidence**

Three equal panels:

### BEFORE

Mock satellite image.

### AFTER

Mock satellite image.

### CHANGE

Same scene with red/yellow highlighted regions.

Use realistic-looking but clearly **demonstration/mock satellite imagery**.

Add a small legend:

**Red — Built-up expansion**

**Yellow — Change area**

Add tiny technical labels such as:

`BEFORE`

`AFTER`

`CHANGE MASK`

Do not add fake numerical measurements.

Do not add percentages.

Do not add fake coordinates pretending to be real measurements.

---

# RESULTS DETAILS

Below the imagery, use compact technical sections.

### Input Details

**Before:** `scene_before.tif`

**After:** `scene_after.tif`

**Status:** Processing completed

### Selected Workflow

**Change-detection model + Change-description model**

### Model / Tool Labels

Use generic labels only:

**Change Detection**

**Change Description**

**Visual Evidence Generator**

Do not invent specific model names.

---

# AUDIT TRAIL

Create a simple expandable section:

**Execution Summary +**

When expanded:

`Query interpreted`

`Inputs validated`

`Task selected`

`Specialist workflow selected`

`Visual evidence generated`

`Answer prepared`

Make it resemble a normal technical log, not an AI explanation card.

---

# THREE DEMO MODES

The same UI structure should work for all three scenarios.

### SINGLE IMAGE

Query:

**Describe the land-cover and major objects visible in this image.**

Answer:

**The image contains agricultural land, a water body, scattered built-up areas, and a road network.**

Workflow:

**Single-image VQA + Scene Description**

Evidence:

**Water / Vegetation / Buildings / Roads**

---

### BI-TEMPORAL CHANGE

Make this the **main demo**.

Query:

**What changed between these two dates, and where did the change occur?**

Answer:

**Built-up area increased in the southern and eastern portions of the scene.**

Workflow:

**Change-detection model + Change-description model**

Evidence:

**Before / After / Change**

Use red/yellow change overlays.

---

### OPTICAL + SAR

Query:

**Use the optical and SAR images together to identify built-up and water-covered regions.**

Answer:

**The fused analysis identifies water-covered regions in the western zone and dense built-up structures in the central-east zone.**

Workflow:

**Cross-modal Optical–SAR Analysis**

Evidence:

**Optical / SAR / Fused Result**

---

# REPORT SCREEN

Keep it simple.

Heading:

**Analysis Report Ready**

Text:

**The demonstration report contains the query, inputs, selected workflow, result, and visual evidence.**

Show a simple report preview.

Button:

**Download Analysis Report**

Secondary:

**New Analysis**

Clicking download can display:

**Demo report download triggered**

---

# MICRO-INTERACTIONS

Use subtle interactions:

* Buttons change slightly on hover
* Selected analysis mode gets a border
* Upload card changes after selecting a demo file
* Validation checks appear progressively
* Workflow steps change from pending → completed
* Expand/collapse audit trail
* Prompt chips populate the query field
* Results update based on selected analysis mode

Keep animations short and subtle.

No flashy animations.

---

# HUMAN-MADE DETAILS

Add small details that make the prototype feel genuinely built rather than AI-generated:

* Slightly varied card sizes based on content
* Practical labels
* Small technical metadata
* Compact spacing
* Simple icons
* Clear empty states
* Normal button sizes
* Realistic filenames
* Small “Demo Mode” labels
* Consistent but not excessive rounded corners
* Functional-looking states
* Clear error/success states
* Sensible alignment rather than decorative symmetry everywhere

The design should feel like someone actually thought about **how a user would operate the tool**, not just how to make a screenshot look impressive.

---

# FINAL DESIGN TEST

Before finalizing, ask:

**“Could this realistically be a prototype coded by a group of engineering students in HTML/CSS/JS?”**

If the answer looks like a futuristic AI startup landing page, simplify it.

The final UI should communicate:

**Practical + Technical + Clean + Trustworthy + Human-built**

rather than:

**Futuristic + Glossy + AI-generated + Over-designed.**
