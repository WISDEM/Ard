import ard
from pathlib import Path

ard_root_dir = Path(ard.__file__).parents[1]
stylesheet_dir = Path(ard_root_dir, "assets", "stylesheets")
stylesheet_seaborn_base_name = "stylesheet_seaborn.mplstyle"
stylesheet_cvf_base_name = "stylesheet_ard_tex.mplstyle"
stylesheet_cvf_notex_name = "stylesheet_ard_notex.mplstyle"


def get_stylesheets(
    dark=False, seaborn_base=True, ard_base=True, use_latex=False,
):
    out_list = []

    # use the matplotlib color scheme by default, dark if specified
    if dark:
        out_list.append("dark_background")

    # building up, use the seaborn base settings I extracted from their github
    if seaborn_base:
        out_list.append(Path(stylesheet_dir, stylesheet_seaborn_base_name))

    # next, apply cory's custom changes
    if ard_base and use_latex:
        out_list.append(Path(stylesheet_dir, stylesheet_cvf_base_name))
    elif ard_base:
        out_list.append(Path(stylesheet_dir, stylesheet_cvf_notex_name))

    return out_list  # kick out the result
