# Sentinel-1 SAR Reading Cheat-Sheet

The rendered channels are fixed-scale gamma-0 backscatter: **red = VV**, **green = VH**, and **blue = VV−VH**. Brightness represents returned microwave energy, not visible color. Compare patterns, texture, context, and geometry; do not treat any single color as a guaranteed class.

## Backscatter physics

| Surface or mechanism | Typical appearance | Physical reason |
|---|---|---|
| Calm open water | Dark | Specular reflection directs energy away from the side-looking radar. Wind or waves can brighten water. |
| Urban/built-up | Bright, angular, heterogeneous | Wall–ground double bounce and strong corner-like reflectors return energy toward the sensor. |
| Forest/dense vegetation | Medium to bright, textured; VH often stronger than bare surfaces | Branches, trunks, and canopy create volume scattering and depolarization. |
| Cropland | Variable, often patterned by field boundaries | Moisture, roughness, crop structure, row direction, and growth stage all affect backscatter. |
| Bare soil | Variable from dark to bright | Roughness and dielectric response from soil moisture dominate; smooth dry soil is often darker. |
| Flooded vegetation | Often unexpectedly bright | Water–stem or water–trunk double bounce can return more energy than either calm water or vegetation alone. |

## Side-looking geometry glossary

- **Near range / far range:** Near range is closer to the radar ground track; far range is farther away. SAR measures slant range, so viewing geometry differs from a map or optical nadir image.
- **Foreshortening:** A slope facing the radar is compressed because its top and bottom have similar slant ranges. The slope appears shortened and often bright.
- **Layover:** A steep radar-facing slope causes the top to return before the base, reversing their apparent order. This is a geometric displacement, not a land-cover boundary.
- **Radar shadow:** Terrain blocks illumination behind a steep slope, producing a dark region with no return. Unlike calm water, shadow follows terrain and look direction.
- **Double bounce:** Two reflections—commonly wall then ground, or trunk/stem then water—send a strong return back toward the radar.
- **Speckle:** Coherent interference creates granular brightness variation. Filtering reduces it but also softens fine structure; interpret spatial patterns rather than isolated pixels.
- **No optical analogue:** SAR brightness depends on wavelength-scale roughness, moisture/dielectric properties, polarization, and look geometry. It is not a grayscale photograph.
