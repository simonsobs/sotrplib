Map matching of transient candidates
====================================

The optics tubes (arrays) of the LAT observe at the same time. Each tube and
band gives a separate depth-1 map. A real transient appears in more than one
of these maps. A glitch or an artifact usually appears in only one map.

The map matcher uses this difference. After the runner analyzes all the maps
of a run, the map matcher does these steps:

1. It groups the transient candidates that are at the same position in maps
   of the same observation.
2. It keeps a group as transient candidates only if a minimum number of
   different arrays detected it, and if one detection has a high SNR.
3. It gives each group a significance and a rank.

The library code is in `sotrplib/sifter/map_matching.py`. The config model is
in `sotrplib/config/map_matching.py`.


When the map matcher runs
-------------------------

The map matcher needs the candidates of all the maps at the same time. Thus,
`BaseRunner` (`sotrplib/handlers/base.py`) does not write the source outputs
of a map when the analysis of the map is complete. The runner does these
steps:

1. It analyzes each map and keeps the result (a `MapResult`) in memory.
2. It gives all the results to the map matcher.
3. It writes the source outputs of each map, with the results of the map
   matching.

The two runners (`basic` and `prefect`) do these steps. Use the prefect
runner to analyze many maps in parallel. See
[Run `sotrp` with the prefect runner](prefect.md).


Configure the map matcher
-------------------------

Add a `map_matcher` field to the `sotrp` config:

```json
"map_matcher": {
    "matcher_type": "multi_array",
    "radius": "1.5 arcmin",
    "min_arrays": 2,
    "high_sig": 5.0,
    "low_sig": 3.0,
    "summary_directory": "/path/to/outputs/"
}
```

| Field | Default | Meaning |
|---|---|---|
| `matcher_type` | `empty` | `empty`: no map matching. `multi_array`: the map matcher on this page. |
| `radius` | `1.5 arcmin` | The maximum distance between two detections of one event. |
| `min_arrays` | `2` | The minimum number of different arrays that must detect an event. |
| `high_sig` | `5.0` | A confirmed event must have one detection with SNR `high_sig` or more. |
| `low_sig` | `3.0` | The map matcher uses only the candidates with SNR `low_sig` or more. |
| `summary_directory` | none | If set, the map matcher writes a JSON summary of the groups in this directory. |

`low_sig` must not be more than `high_sig`.

With `matcher_type: "empty"` (the default), the transient candidates do not
change. Use this value to analyze one map at a time.

**Put all the maps that you want to compare in one run.** The map matcher
compares only the maps of one run. To compare the bands, the run must
contain all the bands. See "Analyze all the maps of one day in one job" in
[Run `sotrp` with the prefect runner](prefect.md).


How the map matcher makes groups
--------------------------------

1. **Observations.** If the time ranges of two maps overlap, the maps are
   in the same observation. The map matcher compares only the candidates of
   maps in the same observation.
2. **Links.** In one observation, the map matcher links two candidates of
   different maps if the distance between them is not more than `radius`.
   It uses only the candidates with SNR `low_sig` or more. The other
   candidates go to `unconfirmed_transient_candidates`, with no `map_match`.
3. **Groups.** Linked candidates are in the same group. The links make
   chains. For example, if A links to B and B links to C, then A, B and C
   are in one group.
4. **Confirmation.** A group is confirmed if both of these conditions are
   true:
   - It contains detections from `min_arrays` or more different arrays. One
     array in two bands counts as one array. The two bands of one tube use
     the same optics, so a problem in the tube can appear in the two bands.
   - One of its detections has SNR `high_sig` or more.

Each matched candidate is in exactly one group. A candidate that has no link
is a group with one member.


Use a low blind-search threshold
--------------------------------

With `high_sig` and `low_sig`, a detection at 5 sigma in one array can be
confirmed by a detection at 3 sigma or more in a different array. To get the
3-sigma detections, set the blind search and the sifter to `low_sig`:

```json
"blind_search": {
    "search_type": "photutils",
    "parameters": {"sigma_threshold": 3.0}
},
"sifter": {
    "sifter_type": "default",
    "cuts": {"snr": [3.0, "inf"]}
},
"map_matcher": {
    "matcher_type": "multi_array",
    "high_sig": 5.0,
    "low_sig": 3.0
}
```

The sifter cuts that you do not give keep their default values (see
`DEFAULT_SIFTER_CUTS` in `sotrplib/sifter/core.py`). If the sifter `snr` cut
is more than `low_sig`, the sifter moves the low-SNR detections to
`noise_candidates`, and the map matcher does not use them.

**Use this configuration only with the `multi_array` map matcher.** With the
`empty` map matcher, all the 3-sigma detections stay transient candidates.

The cost of a 3-sigma blind search. These values are from two f090 i1
depth-1 maps of 2025-09-10. Each map has approximately 850 deg² of data:

| | 5 sigma | 3 sigma |
|---|---|---|
| Blind-search detections in each map | 1 to 8 | approximately 1500 |
| Time for each map | approximately 95 s | approximately 120 s to 135 s |
| Size of the pickle output of each map | approximately 3 MB | approximately 30 MB |

Each detection gets a thumbnail (approximately 15 ms and 18 kB for each
detection). The runner keeps the results of all the maps in memory until
the map matching is complete. Thus, for large maps, examine the memory.

The blind search does not fit a Gaussian to a detection. The `fwhm` of a
detection comes from the second moments of the pixels above the threshold.
Near the threshold, this value has a large error. The sifter `fwhm` cut can
thus remove real low-SNR detections.


Significance and rank
---------------------

The map matcher gives each group a significance. The significance uses the
number of maps in the group and the SNR in each map:

1. `combined_snr` is the square root of the sum of the squared SNRs:
   `combined_snr = sqrt(sum(snr_i**2))`.
2. If each map contains only independent Gaussian noise, `combined_snr**2`
   has a chi-squared distribution with `n_maps` degrees of freedom.
3. The map matcher calculates the probability that noise gives a value of
   `combined_snr**2` or more. Then it changes this probability to a
   one-sided Gaussian significance in sigma.

The significance increases with the number of maps and with the SNR in each
map. For example:

| SNR in each map | `combined_snr` | `significance` |
|---|---|---|
| 11 | 11.0 | 10.9 |
| 5, 5 | 7.1 | 6.7 |
| 6, 6 | 8.5 | 8.1 |
| 64.8, 56.7, 52.8, 51.5, 25.7, 9.7 | 116.6 | 116.4 |

For a very bright event, the chi-squared probability is too small for a
floating-point number. The map matcher then uses the asymptotic form of the
upper incomplete gamma function, so the significance stays finite.

The map matcher ranks the groups of a run in three sets:

1. The confirmed groups.
2. The other notable groups. A notable group has one detection with SNR
   `high_sig` or more, or detections from `min_arrays` or more arrays.
3. All the other groups. These are detections in one array with SNR less
   than `high_sig`.

The map matcher sorts each set by significance. Rank 1 is the most
significant confirmed group. Thus, a bright detection in one array cannot
have a higher rank than a confirmed group.

**Use the significance only to compare the groups.** The significance
assumes that the noise in each map is Gaussian and independent. Glitches and
artifacts are not Gaussian. Also, the map matcher does not use the maps that
cover the position but did not detect the event.


Results
-------

The map matcher adds its results to the candidates. The source outputs (for
example, the pickle files) contain these results.

### The sifter result

The map matcher changes the `SifterResult` of each map:

- `transient_candidates` contains only the candidates of confirmed groups.
- `unconfirmed_transient_candidates` contains the other candidates. The map
  matcher does not delete them.

### The `map_match` field

Each transient candidate (confirmed or not) has a `map_match` field. It is a
`MapMatch` (`sotrplib/sources/sources.py`) with these fields:

| Field | Meaning |
|---|---|
| `match_id` | The identifier of the group. All the candidates of a group have the same `match_id`. |
| `confirmed` | `True` if `min_arrays` or more arrays detected the group, and one detection has SNR `high_sig` or more. |
| `max_snr` | The highest SNR of the detections in the group. |
| `n_maps` | The number of maps with a detection in the group. |
| `n_arrays` | The number of different arrays in the group. |
| `n_bands` | The number of different bands in the group. |
| `combined_snr` | `sqrt(sum(snr_i**2))` of the detections in the group. |
| `significance` | The significance of the group, in sigma. |
| `rank` | The rank of the group in the run. Rank 1 is the most significant. |

### The summary file

If `summary_directory` is set, the map matcher writes one JSON file for each
run. The name of the file is `map_match_summary_<start>.json`. `<start>` is
the earliest start time of the maps of the run. The file contains:

- `min_arrays`, `high_sig`, `low_sig`, `radius_arcmin` and the names of the
  maps of the run;
- `n_groups`: the number of groups;
- `groups`: the confirmed and the notable groups, in the sequence of their
  rank;
- `n_groups_not_listed`: the number of the other groups. The file does not
  list these groups, because a low blind-search threshold can give many of
  them.

Each group has the fields of `MapMatch`, and also:

- `ra_deg` and `dec_deg`: the mean position of the detections, with the
  weight `snr**2`;
- `arrays` and `bands`;
- `members`: one item for each detection, with the map name, `mapcat_id`,
  array, band, SNR, flux, position and time.


Example
-------

This example is one day of lat-iso depth-1 maps (2025-09-10): 24 maps, 6
arrays and 4 bands, in one prefect job. The config has
`"min_arrays": 2`.

| Rank | Confirmed | Significance | Maps | Arrays | Bands | RA, Dec (deg) |
|---|---|---|---|---|---|---|
| 1 | yes | 116.4 | 6 | c1, i1, i3 | f090, f150, f220, f280 | 348.348, +2.676 |
| 2 | no | 16.7 | 1 | i4 | f150 | 10.500, +6.697 |
| 3 | no | 11.9 | 1 | i6 | f150 | 37.150, −6.677 |
| 4 | no | 10.9 | 1 | i3 | f090 | 7.744, −9.500 |
| 5 | no | 6.6 | 1 | i5 | f280 | 40.361, −10.101 |
| 6 | no | 5.8 | 1 | c1 | f220 | 8.702, −8.417 |
| 7 | no | 5.7 | 1 | c1 | f220 | 52.494, +5.207 |

Three arrays detected the event of rank 1 in all four bands. Only one array
detected each of the other events.


Use the map matcher in the library
----------------------------------

You can use the map matcher without a command:

```python
from astropy import units as u

from sotrplib.sifter.map_matching import MultiArrayMapMatcher

matcher = MultiArrayMapMatcher(radius=1.5 * u.arcmin, min_arrays=2)
results, groups = matcher.match(map_results)
```

`map_results` is a list of `MapResult`. `match()` returns the updated
results and the groups (`MapMatchGroup`). Use the returned results, not the
input list. `map_match_significance()` calculates the significance of a list
of SNRs.


Limits
------

- The map matcher does not use the maps that cover the position of an event
  but did not detect it.
- The map matcher does not compare the fluxes of the detections.
- `radius` is the same for all the bands. The beam is smaller at the higher
  bands.
- The map matcher compares only the maps of one run.
