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
  Class: Intersecting dark strips consistent with a runway pattern
  Reasoning: runways are large, flat, smooth paved surfaces that act as specular reflectors (similar to calm water). Radar energy hits them and deflects away from the sensor rather than back to it, producing very low backscatter and rendering them distinctly dark despite being situated in the middle of a high-backscatter urban area.

Vegetation:
No clearly distinct vegetation signature identified in this render. The
urban texture (mixed pink/light-purple, high heterogeneity) dominates
the scene; if parks or tree cover are present, they aren't separable
from the surrounding built-up backscatter at this resolution/composite.

Terrain artifacts (layover/foreshortening/shadow):
No layover, foreshortening, or radar shadow is confidently identifiable in
this render. This does not establish that the terrain is physically flat or
that geometric distortion is absent; the render alone does not provide that
information.

Why it looks this way:
The scene splits into three backscatter regimes driven by surface
roughness and geometry, not color: calm sea/creek water (specular,
dark), dense urban fabric (double-bounce, bright and heterogeneous),
and isolated smooth features — the runway-like strips and an enclosed
dark-water candidate — that both read dark like open water despite sitting
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
  Shape: Rounded/irregular blob outline consistent with pond-like geometry.
  Mechanism: Specular reflection off calm open water (SAR returns very low backscatter
  from flat water surfaces, appearing dark). The enclosed dark patch has pond-like geometry.
  Confidence: Medium — texture and shape support a water interpretation, but
  do not conclusively distinguish water from other low-return surfaces.

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
- Scene: maharashtra_farmland
  Feature: Cropland / low-vegetation mosaic
  Location in frame: Across most of the frame outside the isolated dark water
  bodies and bright clustered patches, especially throughout the upper half and
  between the central and lower built-up clusters.
  Brightness/texture: Predominantly medium violet/lavender with fine granular
  mottling and broad, subtle changes in tone. The texture is less uniformly
  dense and continuous than the canopy signature in the Western Ghats scene.
  Shape: Irregular to weakly rectilinear patchwork with faint parcel-like
  boundaries and elongated field-scale patches, rather than one continuous
  forest canopy.
  Mechanism: A mixture of surface scattering from soil and volume scattering
  from low vegetation. Differences in surface roughness, moisture, vegetation
  structure, and field orientation can all alter the return; this composite
  does not support assigning those causes to individual parcels or identifying
  a crop species.
  Confidence: Medium — the repeated parcel-scale pattern supports managed
  fields or low vegetation, but planted vegetation and exposed soil are not
  reliably separable everywhere in this render.

Terrain artifacts (layover/foreshortening/shadow):
- Scene: maharashtra_farmland
  Feature: No confidently identifiable layover, foreshortening, or radar shadow
  Location in frame: No discrete terrain-artifact region is identifiable across
  the frame.
  Brightness/texture: There are no clear adjacent bright compressed slopes and
  directionally consistent dark shadow zones. The isolated solid dark patches
  are rounded or irregular and fit the visible water signature more closely
  than terrain shadow.
  Shape: No repeated ridge/valley banding, slope compression, or paired
  bright-side/dark-side geometry is resolved strongly enough to make a terrain
  call.
  Mechanism: Layover, foreshortening, and radar shadow depend on slope and radar
  look direction; the required geometric relationships are not evident here.
  RTC processing does not by itself remove those distortions, so their absence
  from this annotation means only that they cannot be confidently identified
  in this crop.
  Confidence: Medium-high that no artifact is identifiable, not that geometric
  distortion is physically absent.

Why it looks this way:
The scene is dominated by a field-scale mosaic whose varying medium
backscatter is consistent with changing surface roughness, moisture, soil
exposure, vegetation structure, and field orientation; the SAR image alone
does not distinguish those factors parcel by parcel or identify crop species.
Bright, heterogeneous magenta clusters are consistent with strong returns from
hard structures and wall-ground double bounce, while the isolated smooth
dark-navy ponds return little energy because calm water reflects it away from
the sensor. Fine granular variation across the scene is also consistent with
SAR speckle. No additional brightness pattern is sufficiently tied to slope
and look geometry to justify a layover, foreshortening, or radar-shadow claim.

## Western Ghats forest

![Western Ghats forest RTC false color](rendered/western_ghats_forest.png)

Water areas:
- Region: sinuous, narrow dark-navy channel running through the upper-central
  portion of the frame, branching into smaller tributaries
  Class: water (river/stream)
  Reasoning: The dark, sinuous, channel-like feature is consistent with water;
  flow state is not determinable from a single static render; the
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
  main channel. Possible minor tributaries or small water candidates. Confidence: Low
  — too small to rule out speckle/shadow at this scale.

Urban/built-up:
- No confidently identifiable urban cluster. A handful of faint, isolated
  brighter pixels sit along what may be trail or ridge lines, but they are
  sparse, unclustered, and equally consistent with small exposed rock
  outcrops or isolated structures. SAR alone does not disambiguate these at
  this resolution — genuinely uncertain, not forcing a settlement call.

Vegetation:
- Region: dominates essentially the entire valid-data portion of the frame
  (left ~55%; the black region on the right is the scene's nodata/no-coverage
  region, not a land-cover class)
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
right is a nodata/no-coverage region, not a physical
land-cover feature, and should not be read as radar shadow.

## Konkan coast

![Konkan coast RTC false color](rendered/konkan_coast.png)

Water areas:
- Scene: konkan_coast
  Feature: Open coastal water
  Location in frame: Broad continuous area covering approximately the left half
  of the frame, bounded on the right by the irregular coastline.
  Brightness/texture: Predominantly dark violet/navy with low internal contrast
  and only subtle broad mottling; markedly smoother and lower-return than the
  adjoining land.
  Shape: Large continuous region with an irregular land-water boundary rather
  than an enclosed blob, straight paved strip, or data-coverage edge.
  Mechanism: Specular reflection from a comparatively smooth open-water surface
  directs most radar energy away from the sensor. The small tonal variations can
  reflect changes in surface roughness, but this render does not support assigning
  a specific cause to them.
  Confidence: High — the extent, low-return texture, and connected coastline
  geometry support open coastal water.
- Scene: konkan_coast
  Feature: Estuary/inlet and connected river channels
  Location in frame: A narrow, sinuous dark channel enters from the upper-center
  coast and winds across the upper-right land area; a second branching dark
  channel system meets the coast near the lower-center and extends into the
  lower-right.
  Brightness/texture: Dark navy to violet and relatively uniform within the
  channels, with substantially less granular backscatter than the surrounding
  land.
  Shape: Meandering, branching paths that remain continuous with the broad
  coastal water. Their drainage-like geometry argues against smooth pavement,
  isolated nodata, or a terrain shadow unrelated to the shoreline.
  Mechanism: Low backscatter is consistent with specular reflection from smooth
  water in tidal inlets, an estuary, or river channels.
  Confidence: High that the connected dark features are water; Medium-high on
  the more specific estuary/inlet interpretation because flow direction and
  tidal state are not recoverable from this image alone.

Urban/built-up:
- Scene: konkan_coast
  Feature: Sparse possible built structures or other hard-surface returns; no
  confidently identifiable dense urban cluster
  Location in frame: Small isolated bright pink-to-white specks occur on the
  land side, including near the lower-central inlet and at scattered inland
  locations to the right.
  Brightness/texture: A few high-return pixels and tiny rough patches are visible,
  but they do not form the broad, bright, heterogeneous texture seen in the
  Mumbai urban area.
  Shape: Sparse and weakly clustered, without a resolved street grid, dense block,
  or sufficiently continuous angular pattern. Individual returns could equally
  arise from exposed rock, rough ground, or isolated hard structures.
  Mechanism: Strong local returns can be produced by wall-ground double bounce
  or corner-like reflectors, but brightness alone is not enough to identify a
  building in this scene.
  Confidence: Low for identifying the scattered returns as built-up; High that
  no dense urban signature is confidently resolved in this crop.

Vegetation:
- Scene: konkan_coast
  Feature: Vegetated or rough land-cover mosaic
  Location in frame: Most of the land east of the coastline, including the
  peninsula-like area between the upper channel and the sea and the inland areas
  between the branching waterways.
  Brightness/texture: Mainly medium lavender/violet with fine mottling and broad
  patch-to-patch tonal variation. It is more textured and higher-return than the
  adjacent water but is not uniformly bright or continuous enough to assign the
  entire land area to dense forest.
  Shape: Irregular, contiguous patches following the land-water boundary, with
  some weak internal patchiness but no consistently resolved row pattern or
  parcel geometry across the scene.
  Mechanism: The land return is consistent with a mixture of volume scattering
  from vegetation and surface scattering controlled by roughness and moisture.
  This composite cannot separate those contributions for individual patches or
  identify vegetation type or crop species.
  Confidence: Medium that vegetation contributes substantially to the mottled
  land signature; Low for distinguishing continuous canopy, low vegetation, and
  exposed soil within individual patches.

Terrain artifacts (layover/foreshortening/shadow):
- Scene: konkan_coast
  Feature: No confidently identifiable layover, foreshortening, or radar shadow
  Location in frame: No discrete terrain-artifact region is resolved across the
  coastal land or along the inland channel systems.
  Brightness/texture: The scene lacks a systematic pairing of compressed bright
  slopes with adjacent, directionally consistent dark shadow zones. Narrow dark
  segments near the coast and channels are continuous with plausible water
  geometry and are not independently diagnostic of terrain shadow.
  Shape: No repeated ridge-aligned bands, slope reversals, or bright-side/dark-side
  pairs are clear enough to tie to radar look direction.
  Mechanism: Coastal relief could modulate local incidence angle and backscatter,
  but the necessary viewing-geometry relationship is not identifiable here. RTC
  does not remove layover or foreshortening, so this is an identifiability finding,
  not a claim that geometric distortion is physically absent.
  Confidence: Medium-high that no discrete terrain artifact can be confidently
  identified in this render.

Why it looks this way:
The broad coastal water and connected sinuous channels are dark because smooth
water produces predominantly specular reflection, directing little energy back
to the side-looking radar. The land is brighter and finely mottled because
vegetation volume scattering, surface roughness, moisture, and coherent speckle
all contribute to spatially variable backscatter; the image does not isolate
those causes patch by patch. Sparse very bright returns may come from isolated
hard structures and double-bounce geometry, but their limited extent and
ambiguous shape do not support a dense urban classification. Although coastal
relief may affect local incidence angle, no brightness-darkness pattern is tied
strongly enough to slope and look direction to label layover, foreshortening, or
radar shadow in this crop.

## Flat inland plain

![Flat inland plain RTC false color](rendered/flat_inland_plain.png)

Water areas:
- Scene: flat_inland_plain
  Feature: Possible narrow drainage channel and small ponds
  Location in frame: A narrow, meandering dark line runs near the lower edge
  through the rectilinear parcel area; several compact dark-navy patches are
  scattered across the central and right-hand portions of the frame.
  Brightness/texture: The candidate features have low backscatter and little
  visible internal mottling compared with the surrounding land. Some of the
  smallest patches approach the scale of local speckle and cannot be classified
  from tone alone.
  Shape: The lower feature follows a continuous curved path compatible with a
  small channel, while the compact candidates have irregular or rounded outlines
  compatible with ponds. In contrast, the large dark blocks at the upper-left
  and lower-right have straight parcel boundaries and internal linework, so they
  are not treated as open water solely because they are dark.
  Mechanism: Smooth water can return little energy through specular reflection.
  Smooth soil, pavement, or other low-roughness surfaces can also appear dark,
  while radar shadow is less plausible without an associated slope geometry.
  Confidence: Medium for the meandering feature as a water channel; Low-to-medium
  for the isolated pond candidates because their small size and ambiguous context
  do not rule out smooth land surfaces or speckle.

Urban/built-up:
- Scene: flat_inland_plain
  Feature: Sparse candidate built-up clusters or other hard-surface features
  Location in frame: Bright pink-to-white speckled clusters occur along a loose
  diagonal from the lower-left toward the central portion of the frame, with
  smaller clusters and isolated bright returns farther right and toward the
  upper-middle.
  Brightness/texture: The strongest areas are locally bright and heterogeneous,
  containing near-saturated specks within rough patches. They are separated by
  broad areas of lower-return field texture rather than forming one continuous
  urban mass.
  Shape: Small irregular clusters, some loosely aligned with narrow linear
  features. The pattern is compatible with dispersed structures, but isolated
  bright pixels and short bright lines do not resolve individual buildings or a
  street grid.
  Mechanism: Wall-ground double bounce and corner-like reflections from hard
  structures can produce these strong returns. Exposed rock, rough bare ground,
  or other hard surfaces remain alternatives where the returns are isolated.
  Confidence: Medium for built-up or hard structures in the brightest clustered
  areas; Low for assigning the scattered individual returns specifically to
  buildings.

Vegetation:
- Scene: flat_inland_plain
  Feature: Field and low-vegetation mosaic
  Location in frame: Dominates most of the frame outside the bright candidate
  structural clusters and the small dark features, with especially clear
  rectilinear blocks in the upper-left and lower-right and finer patchwork across
  the center.
  Brightness/texture: Mostly medium lavender/violet with fine mottling and broad
  parcel-to-parcel changes in tone. Some large parcels are smoother and darker,
  while others are brighter or more granular; no uniformly textured continuous
  canopy dominates the scene.
  Shape: Repeated rectangular and elongated boundaries, internal parallel lines,
  and adjoining irregular patches form a managed field-scale pattern rather than
  a single uninterrupted forest signature.
  Mechanism: A varying mixture of surface scattering from soil and volume
  scattering from low vegetation can explain the parcel-scale contrast. Surface
  roughness, moisture, vegetation structure, and field orientation may each
  affect the return, but this image cannot assign a cause to every parcel or
  identify crop species.
  Confidence: Medium-high that the repeated boundary-defined pattern is a field
  or low-vegetation mosaic; Low for separating vegetation from exposed soil
  within individual parcels.

Terrain artifacts (layover/foreshortening/shadow):
- Scene: flat_inland_plain
  Feature: No confidently identifiable layover, foreshortening, or radar shadow
  Location in frame: No discrete terrain-artifact region is identifiable across
  the frame.
  Brightness/texture: There are no adjacent compressed bright slopes and
  directionally consistent dark shadow zones. Dark regions are instead compact,
  channel-like, or bounded by the rectilinear land-cover pattern.
  Shape: No repeated ridge-aligned bands, displaced slope features, or paired
  bright-side/dark-side geometry is resolved strongly enough to establish a
  terrain effect tied to radar look direction.
  Mechanism: Layover, foreshortening, and radar shadow require slope and viewing
  geometry that are not evident in this render. RTC does not remove layover or
  foreshortening, so this annotation means only that no artifact can be
  confidently identified, not that geometric distortion is physically absent.
  Confidence: High that no discrete terrain artifact is identifiable in this
  crop.

Why it looks this way:
The scene is dominated by a boundary-defined field mosaic whose varying medium
backscatter is consistent with changing surface roughness, moisture, exposed
soil, low vegetation structure, and field orientation. Smooth, dark parcels are
not automatically water because their straight boundaries and internal linework
also support managed land surfaces; only the meandering and compact low-return
features have plausible channel or pond geometry, and the smaller candidates
remain uncertain. Bright heterogeneous clusters may reflect strong returns from
hard structures and wall-ground double bounce, whereas isolated bright pixels
are not diagnostic by themselves. Fine granular variation is consistent with
SAR speckle. No systematic bright-slope/dark-shadow relationship is visible, so
the image does not support a layover, foreshortening, or radar-shadow call.
