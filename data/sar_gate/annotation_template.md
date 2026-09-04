# SAR Reading Exercise — Manual Annotations

Read `sar_cheatsheet.md` first. Fill every field manually from the rendered SAR image; do not infer from optical maps or automated labels.

## Mumbai coastal

![Mumbai coastal RTC false color](rendered/mumbai_coastal.png)

- **Location:** 19.05, 72.85
- **Job ID:** 71cf874e-4303-4e1f-9ab7-74037b1956c9

Water areas:
- Region: large dark-blue area, left side and water inlets (right/bottom)
  Class: water
  Reasoning: calm sea/creek water acts as a smooth surface — radar energy reflects away from the sensor (specular reflection) rather than back to it, so almost no signal returns and it renders dark. Slight brightness variation in some patches likely reflects rougher water or shallower/turbid areas scattering a bit more energy back.

Urban/built-up:
- Region: large bright pink/white textured area covering most of the middle-to-right (Mumbai city)
  Class: urban/built-up
  Reasoning: dense construction creates double-bounce geometry — radar energy strikes vertical walls and horizontal ground/streets at right angles (corner reflector effect), returning a large portion of energy back to the sensor. The speckled/textured appearance reflects varying building heights, orientations, and street canyons.
- Region: dark X-shaped feature in the center of the bright urban area
  Class: airport runway (Chhatrapati Shivaji Maharaj International Airport)
  Reasoning: runways are large, flat, smooth paved surfaces that act as specular reflectors (similar to calm water). Radar energy hits them and deflects away from the sensor rather than back to it, producing very low backscatter and rendering them distinctly dark despite being situated in the middle of a high-backscatter urban area.

Vegetation:
No clearly distinct vegetation signature identified in this render. The
urban texture (mixed pink/light-purple, high heterogeneity) dominates
the scene; if parks or tree cover are present, they aren't separable
from the surrounding built-up backscatter at this resolution/composite.

Terrain artifacts (layover/foreshortening/shadow):
None observed. Mumbai's coastal terrain is largely flat, so there is no
steep-slope geometry to produce layover, foreshortening, or radar
shadow.

Why it looks this way:
The scene splits into three backscatter regimes driven by surface
roughness and geometry, not color: calm sea/creek water (specular,
dark), dense urban fabric (double-bounce, bright and heterogeneous),
and isolated smooth features — the airport runway and an inland lake
(Powai/Vihar) — that both read dark like open water despite sitting
inside or beside the urban mass, because they're flat surfaces that
reflect radar energy away from the sensor. Brightness alone doesn't
distinguish these dark features from each other; shape and context
(linear runway vs. enclosed lake blob vs. irregular coastline) does.

## Maharashtra farmland

![Maharashtra farmland RTC false color](rendered/maharashtra_farmland.png)

Water areas:
- Scene: maharashtra_farmland
  Feature: Water area (pond)
  Location in frame: mid-left, isolated in open farmland texture, not touching the
  built-up/pink cluster
  Brightness/texture: Solid, flat, uniform dark navy tone. No internal speckle or
  mottling — matches the calm-water signature seen in the Mumbai coastal scene.
  Shape: Rounded/irregular blob outline. No straight edges detected — rules out
  canal, bund, or engineered irrigation tank.
  Mechanism: Specular reflection off calm open water (SAR returns very low backscatter
  from flat water surfaces, appearing dark). Natural farm pond / seasonal puddle,
  not a constructed water feature, based on absence of straight boundaries.
  Confidence: High — texture and shape both consistent with natural still water,
  same mechanism as prior Mumbai water annotation.

Urban/built-up:
- Scene: maharashtra_farmland
  Feature: Urban/built-up area (settlement, uneven density)
  Location in frame: Central-left region, spanning from middle-left down to bottom-left
  Brightness/texture: Bright pink/magenta with rough, speckled internal texture,
  including scattered near-saturated bright specks — consistent with strong
  double-bounce backscatter off buildings and hard structures. Density is uneven:
  the middle-left portion is bright-dominant (dense core), while the bottom-left
  portion is dark-dominant with only sparse bright pixels (thinning periphery).
  Shape: Irregular clustered patches loosely following a linear road/path feature
  visible running through the region, rather than a dense block or planned grid —
  consistent with a rural settlement strung along a road, with density fading
  outward into scattered structures.
  Mechanism: Double-bounce scattering (ground-to-wall-to-sensor) off buildings and
  vertical hard structures. Note: brightness/roughness alone cannot distinguish
  settlement from other hard-surface features (e.g. rock outcrops, industrial
  structures) from SAR signature alone — this interpretation is based on pattern
  and rural context, not confirmed against optical imagery or map data.
  Confidence: Medium — texture and mechanism are consistent with built-up area, but
  identification as settlement (vs. other hard-surface feature) is inferred from
  context, not independently verified.

Vegetation:

Terrain artifacts (layover/foreshortening/shadow):

Why it looks this way:

## Western Ghats forest

![Western Ghats forest RTC false color](rendered/western_ghats_forest.png)

Water areas:
- Region: sinuous, narrow dark-navy channel running through the upper-central
  portion of the frame, branching into smaller tributaries
  Class: water (river/stream)
  Reasoning: near-zero backscatter following a continuous dendritic path is
  consistent with specular reflection off a flowing water surface; the
  branching, drainage-shaped geometry (not a straight line, not adjacent to
  an implied steep slope) argues for a stream channel rather than radar
  shadow. Confidence: High.
- Region: irregular, branching dark-navy blob in the lower-left of the frame
  Class: water (pond/reservoir)
  Reasoning: same near-zero backscatter and specular signature as the
  channel above, but as an enclosed blob rather than a linear feature —
  consistent with a small reservoir or natural pool rather than a river.
  Confidence: Medium-high — shape is consistent with standing water, but a
  dammed reservoir vs. a natural pool can't be distinguished from SAR alone.
- Several smaller isolated dark patches scattered through the frame, each a
  few pixels wide, generally aligned along the same drainage pattern as the
  main channel. Likely minor tributaries or seasonal pools. Confidence: Low
  — too small to rule out speckle/shadow at this scale.

Urban/built-up:
- No confidently identifiable urban cluster. A handful of faint, isolated
  brighter pixels sit along what may be trail or ridge lines, but they are
  sparse, unclustered, and equally consistent with small exposed rock
  outcrops or isolated structures. SAR alone does not disambiguate these at
  this resolution — genuinely uncertain, not forcing a settlement call.

Vegetation:
- Region: dominates essentially the entire valid-data portion of the frame
  (left ~55%; the black region on the right is the scene's nodata/swath
  edge, not a land-cover class)
  Class: forest / dense vegetation
  Reasoning: uniform, moderately bright, finely mottled texture across the
  whole scene, consistent with volume scattering from a multi-layered
  canopy (branches, trunks, leaf structure) rather than the flatter,
  boundary-defined patterning seen in the Maharashtra farmland scene.
  Confidence: High — this is the dominant, unambiguous signature here.

Terrain artifacts (layover/foreshortening/shadow):
- No confidently paired bright-slope/dark-slope pattern is identified in
  this crop — i.e., no region where a bright compressed slope sits directly
  beside a corresponding dark shadow zone in a way that clearly indicates
  foreshortening or layover. There is subtle, gradual tonal variation
  across the canopy that could reflect underlying topography modulating
  local incidence angle, but it is not distinct or systematic enough here
  to confidently call out as a discrete terrain artifact. If a wider or
  differently-cropped render of this scene shows clearer ridge/valley
  banding, that should be re-assessed against this note rather than
  assumed present.

Why it looks this way:
The scene is overwhelmingly forest canopy under volume scattering, which
produces the uniform mottled backscatter across nearly the whole valid
swath. Water stands out sharply against this because specular reflection
from calm channels and ponds returns almost no energy to the sensor,
producing near-zero backscatter regardless of the surrounding vegetation
signature. The absence of a clear double-bounce/urban signature is
consistent with a forested, sparsely settled landscape rather than any
claim that no settlement exists — SAR alone can't confirm absence, only
that no strong signature is present in this crop. The black region on the
right is a data-coverage boundary (end of the SAR swath), not a physical
land-cover feature, and should not be read as radar shadow.

## Konkan coast

![Konkan coast RTC false color](rendered/konkan_coast.png)

Water areas:

Urban/built-up:

Vegetation:

Terrain artifacts (layover/foreshortening/shadow):

Why it looks this way:

## Flat inland plain

![Flat inland plain RTC false color](rendered/flat_inland_plain.png)

Water areas:

Urban/built-up:

Vegetation:

Terrain artifacts (layover/foreshortening/shadow):

Why it looks this way:
