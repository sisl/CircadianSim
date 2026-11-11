# House style of the paper figures, matching the figures as assembled for the manuscript (PowerPoint):
# Times Bold axis labels and tick labels, panel names inside the panels, variable names with subscripts.
#
# Labels with subscripts or Greek use GR's TeX-like text (`textext`, which Plots uses when a label contains a
# backslash): `_{...}` gives a subscript and \tau-style commands give Greek, in the current font. A trailing `\ `
# (backslash space) only switches `textext` on, for labels without Greek. `textext` typesets "/" as a fraction, so
# μ_H/χ uses U+2215 (division slash).

const FIG_BOLD = "Times Bold"
const FIG_REGULAR = "Times"

# Print size. PLOS Computational Biology: width ≤ 7.5 in (19 cm), height ≤ 8.75 in, text 8–12 pt at that size.
# Plots measures `size` at 100 units per inch, so with dpi = 300 a canvas of print_size(w, h) is exactly w × h
# inches (3 × that in pixels) and font sizes are the printed point sizes. Figures drawn on larger canvases are
# scaled down to the page, so their text prints at pt × PAGE_WIDTH_IN / (canvas width / 100).
const PAGE_WIDTH_IN, PAGE_HEIGHT_IN = 7.5, 8.75
# Figure heights with room for the caption: half-page figures ≤ 4.25 in, full-page figures ≤ 7.0 in (≥ 1.75 in of
# the page left for the caption)
const HALF_PAGE_H_IN, FULL_PAGE_H_IN = 4.25, 7.0
# Supporting Information figures are placed at the text width of the SI document (\textwidth = 6.5 in)
const SI_WIDTH_IN = 6.5
print_size(w_in, h_in) = (round(Int, 100 * w_in), round(Int, 100 * h_in))
# Printed sizes for figures drawn at print size
const PT_TICK, PT_GUIDE, PT_LABEL, PT_LEGEND = 8, 9, 9, 8

const DISPLAY_LABELS = Dict(
    "S_On" => raw"S_{ON}\ ", "S_Off" => raw"S_{OFF}\ ", "S_Dur" => raw"S_{DUR}\ ", "S_MP" => raw"S_{MP}\ ",
    "CBTmin" => raw"CBT_{MIN}\ ", "CBT_min" => raw"CBT_{MIN}\ ", "ψ_obs" => raw"\psi_{OBS}",
    "CBTmin (Hour)" => raw"CBT_{MIN} (Hour)\ ",   # `\ ` must come last: text after it is dropped
    "τ" => raw"\tau", "K" => raw"\kappa", "k" => raw"\kappa", "κ" => raw"\kappa", "χ" => raw"\chi",
    "μ_H/χ" => "\\mu_{H}\u2215\\chi", "ψ" => raw"\psi", "φ" => raw"\psi",
)

"""Figure label for an analysis variable name (the name itself if it has no special label)."""
display_label(name) = get(DISPLAY_LABELS, String(name), String(name))
