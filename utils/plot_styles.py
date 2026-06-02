# utils/plot_styles.py
"""
Matplotlib style utilities for the FOPT Dark Photon project.

Provides:
  - Color / linestyle / marker registries (blues, reds, colors, linestyles, markers)
  - apply_standard_formatting(ax, ...)  : tick, spine, and axis-label formatting
  - apply_ordered_legend(ax, ...)       : deduplicated legend with lines before markers
  - set_global_style()                  : apply project-wide rcParams once at import
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator

# ------------------------------------------------------------------ #
# Global rcParams                                                       #
# ------------------------------------------------------------------ #
matplotlib.rcParams.update({
    "font.family":              "serif",
    "mathtext.fontset":         "dejavusans",
    "axes.unicode_minus":       False,
    "figure.dpi":               100,
    "pdf.fonttype":             42,
    "text.usetex":              False,
    "axes.formatter.use_mathtext": False,
})

# Computer Modern font (bundled with matplotlib)
_cmfont = font_manager.FontProperties(
    fname=matplotlib.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
matplotlib.rcParams["font.serif"] = _cmfont.get_name()

# ------------------------------------------------------------------ #
# Style registries                                                      #
# ------------------------------------------------------------------ #
blues = ["#313695", "#4575b4", "#74add1"]
reds  = ["#a50026", "#d73027", "#f46d43", "#fdae61"]

custom_labels = {
    "HighT":   r"High-$T$",
    "Full":    r"Full (1L)",
    "OPA":     r"OPA",
    "1L LO":   r"DRalgo: 1L (M-LO)",
    "1L NLO":  r"DRalgo: 1L (M-NLO)",
    "2L M1":   r"DRalgo: 2L (M1)",
    "2L M2":   r"DRalgo: 2L (M2)",
}

linestyles = {
    "HighT":  "--",
    "Full":   "-",
    "OPA":    ":",
    "1L LO":  "-",
    "1L NLO": "--",
    "2L M1":  "-",
    "2L M2":  "--",
}

colors = {
    "Full":    blues[0],
    "HighT":   blues[1],
    "OPA":     blues[2],
    "1L LO":   reds[0],
    "1L NLO":  reds[1],
    "2L M1":   reds[2],
    "2L M2":   reds[3],
}

markers = {
    "1L LO":  "o",
    "1L NLO": "*",
    "2L M1":  "s",
    "2L M2":  "D",
}

marker_colors = {
    "1L LO":  reds[0],
    "1L NLO": reds[1],
    "2L M1":  reds[2],
    "2L M2":  reds[3],
}

legend_markers = {
    "DRalgo: 2L (M1)":   "s",
    "DRalgo: 2L (M2)":   "D",
    "DRalgo: 1L (M-LO)": "o",
    "DRalgo: 1L (M-NLO)":"*",
}

# ------------------------------------------------------------------ #
# Axis formatting                                                       #
# ------------------------------------------------------------------ #

def apply_standard_formatting(
    ax,
    xlog=False, ylog=False,
    xlim=None,  ylim=None,
    xlabel=None, ylabel=None, title=None,
    legend_keys=None, legend_loc="lower right",
):
    """
    Apply the project standard to a matplotlib Axes object.

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
    xlog, ylog  : set log scale and log-spaced minor ticks.
    xlim, ylim  : axis limits as (lo, hi) tuples.
    xlabel, ylabel, title : axis labels / title strings.
    legend_keys : list of keys from the style registries; draws a formatted legend.
    legend_loc  : legend location string.
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(1.5)
        ax.spines[side].set_color("black")

    ax.tick_params(axis="both", which="major", width=1.5, length=8,  direction="in")
    ax.tick_params(axis="both", which="minor", width=1.2, length=4,  direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(labelsize=19)

    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=23)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
    if title:
        ax.set_title(title, fontsize=17)

    if legend_keys:
        handles = [
            Line2D(
                [0], [0],
                color=colors[k], linestyle=linestyles[k],
                linewidth=1.5, label=custom_labels[k],
            )
            for k in legend_keys if k in custom_labels
        ]
        ax.legend(handles=handles, fontsize=13, frameon=False,
                  loc=legend_loc, ncol=2)


# ------------------------------------------------------------------ #
# Ordered legend helper                                                 #
# ------------------------------------------------------------------ #

def apply_ordered_legend(ax, loc="best"):
    """
    Build a deduplicated legend with line handles before marker handles.

    Duplicate labels (same string) are silently dropped, keeping the first
    occurrence.  PathCollection scatter handles are converted to Line2D markers
    using the legend_markers registry.
    """
    handles, labels = ax.get_legend_handles_labels()

    seen     = set()
    filtered = [(h, l) for h, l in zip(handles, labels)
                if not (l in seen or seen.add(l))]

    line_handles, marker_handles = [], []
    line_labels,  marker_labels  = [], []

    for h, l in filtered:
        if isinstance(h, Line2D) and h.get_marker() == "None":
            line_handles.append(h)
            line_labels.append(l)
        elif isinstance(h, Line2D):
            m = Line2D(
                [0], [0],
                marker=h.get_marker(), linestyle="None",
                color=h.get_markerfacecolor(), markersize=8, label=l,
            )
            marker_handles.append(m)
            marker_labels.append(l)
        elif isinstance(h, PathCollection):
            m = Line2D(
                [0], [0],
                marker=legend_markers.get(l, "o"), linestyle="None",
                color=h.get_facecolor()[0], markersize=8, label=l,
            )
            marker_handles.append(m)
            marker_labels.append(l)

    ax.legend(
        line_handles + marker_handles,
        line_labels  + marker_labels,
        fontsize=13, frameon=False, loc=loc, ncol=2,
    )


# ------------------------------------------------------------------ #
# One-shot global style setter                                          #
# ------------------------------------------------------------------ #

def set_global_style():
    """Re-apply project-wide rcParams (idempotent; safe to call multiple times)."""
    matplotlib.rcParams.update({
        "font.family":              "serif",
        "mathtext.fontset":         "dejavusans",
        "axes.unicode_minus":       False,
        "figure.dpi":               100,
        "pdf.fonttype":             42,
        "text.usetex":              False,
        "axes.formatter.use_mathtext": False,
    })
