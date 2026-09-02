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

Terrain artifacts (layover/foreshortening/shadow):

Why it looks this way:

## Maharashtra farmland

![Maharashtra farmland RTC false color](rendered/maharashtra_farmland.png)

Water areas:

Urban/built-up:

Vegetation:

Terrain artifacts (layover/foreshortening/shadow):

Why it looks this way:

## Western Ghats forest

![Western Ghats forest RTC false color](rendered/western_ghats_forest.png)

Water areas:

Urban/built-up:

Vegetation:

Terrain artifacts (layover/foreshortening/shadow):

Why it looks this way:

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
