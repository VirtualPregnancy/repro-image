import inspect

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import sqlite3
import pandas as pd
import copy


def cubify(arr, roi_shape):
    """
    :param arr: array of values, intended to be 3D
    :param roi_shape: list or tuple of values defining the shape of the sub-array that you input array will be reshaped
    into
    :return: a num elems in array/num elems in arrays of shape.roi_shape by num elems in arrays of shape.roi_shape array
    where array[0][:] represents the elements in the 0th sub array of the input
    """
    oldshape = np.array(arr.shape)
    repeats = (oldshape / roi_shape).astype(int) # how many times each dimension of roi_shape repeat for each dimension
    # of the input array
    tmpshape = np.column_stack([repeats, roi_shape]).ravel() # a 1-D array, where the nth odd element are is the repeat
    # number for that dimension and the nth even element is the roi shape corresponding to that dimension
    order = np.arange(len(tmpshape)) # array used for reordering the permuted array in accordance with (repeats,
    # roi_shape)
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    arr = arr.reshape(tmpshape)
    arr = arr.transpose(order) # arr is snow of arr.shape = (repeats, roi_shape)
    arr = arr.reshape(-1, np.prod(roi_shape)) # reshaped to n rois, by n elems in roi
    return arr

def calculate_fractal_dimension(coeff_vars, window_sizes, type='Greyscale', dim=3):
    """
    :param coeff_vars:list of 1D array of floats, each coefficient of variation corresponds to the correesponding window
    size in the window_sizes array
    :param window_sizes:
    :param type: string determining the type of image that was analysed, either greyscale or binary
    :param dim:
    :return:
    """
    if type == 'Greyscale':
        grad, _ = np.polyfit(np.log(window_sizes), np.log(coeff_vars), 1)
        frac_dim = 1 - grad
    elif type == 'binary':

        logx = np.log(window_sizes)
        logy = np.log(coeff_vars)

        m, b = np.polyfit(logx, logy, 1)
        frac_dim = -1 * m
    return frac_dim

def cartesian_product(*arrays):
    """
    :param arrays:
    :return:
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def confirm_directory(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        print(f"{directory} did not exist, it has been created")
    return

def mean_wave(x_values, y_values, verbose=False):
    """
    Compute an average beat waveform from a contiguous Doppler waveform.

    Method summary (aligned with current usseg beat logic):
    1) Propose systolic anchor peaks with prominence-based peak finding and
       merge peaks that are too close.
    2) For each anchor peak, build a backward search window in the preceding
       part of the beat.
    3) In that window:
       - smooth the signal,
       - compute first derivative (slope) and second derivative (change in slope),
       - find a second-derivative anchor (strongest upslope acceleration),
       - walk backward on first derivative to the onset of low slope (foot onset).
    4) Build beats foot-to-foot and derive PS/ED points within each beat for
       diagnostics.
    5) Segment foot-to-foot, align beats to a common x-axis, average, remove
       outlier beats, and recompute the final mean wave.

    :param x_values: numpy array of x values (typically time/sample position).
    :param y_values: numpy array of y values (waveform amplitude/velocity envelope).
    :param verbose: boolean controlling diagnostic plotting output.
    :return: (mean_y, x_common, valid_beats, diagnostics). mean_y and x_common are
        the averaged waveform; valid_beats has shape (n_retained, len(x_common))
        with each retained beat aligned on x_common; diagnostics has beat counts.
    """
    ### 1) Propose anchor peaks, then merge peaks that are too close.
    wave_amplitude = y_values.max()-y_values.min()

    peak_indices, _ = find_peaks(y_values, prominence=wave_amplitude / 4)
    # Min separation (assume x is time): 200 bpm -> 0.3 s; merge peaks closer than that, keep highest
    if len(peak_indices) > 1 and len(x_values) >= 2:
        dx = float(np.median(np.diff(x_values)))
        if np.isfinite(dx) and dx > 0:
            min_distance = max(1, int(60.0 / 200.0 / dx))
            order = np.argsort(peak_indices)
            peaks = peak_indices[order]
            consolidated = []
            i = 0
            while i < len(peaks):
                j = i
                best = int(peaks[i])
                while j + 1 < len(peaks) and (int(peaks[j + 1]) - int(peaks[j])) <= min_distance:
                    j += 1
                    cand = int(peaks[j])
                    if y_values[cand] > y_values[best]:
                        best = cand
                consolidated.append(best)
                i = j + 1
            peak_indices = np.array(consolidated, dtype=int)
    ### 2) For each anchor peak, build a backward search window.
    ### 3) In each window, detect foot onset via derivative logic.
    # second-derivative anchor -> backward first-derivative onset threshold.
    foot_indices = []
    search_windows = []
    debug_rows = []

    search_fraction = 0.50
    smooth_window_max = 11
    polyorder = 2
    min_samples_before_peak = 3
    foot_max_rel_height = 0.55

    for i in range(0, len(peak_indices)):
        peak = int(peak_indices[i])
        if i == 0:
            if len(peak_indices) >= 3:
                diffs = np.diff(peak_indices).astype(float)
                other = diffs[1:] if len(diffs) >= 2 else diffs
                interval_est = int(np.round(np.mean(other))) if other.size > 0 else 0
            elif len(peak_indices) >= 2:
                interval_est = int(peak_indices[1] - peak_indices[0])
            else:
                interval_est = 0
            if interval_est < 5:
                continue
            interval = int(interval_est)
            prev_peak = max(0, peak - interval)
        else:
            prev_peak = int(peak_indices[i - 1])
            interval = int(peak - prev_peak)
        if interval < 5:
            continue

        local_search_fraction = float(search_fraction) if i == 0 else max(0.0, float(search_fraction) - 0.10)
        search_len = max(3, int(local_search_fraction * interval))
        # If first-wave search would extend before signal start, skip this beat.
        if i == 0 and (peak - search_len) < 0:
            continue
        search_start = max(prev_peak, peak - search_len)
        search_end = max(search_start + 2, peak - int(max(1, min_samples_before_peak)))
        if search_end <= search_start + 2:
            continue

        x_region_raw = np.asarray(x_values[search_start:search_end], dtype=float)
        y_region = y_values[search_start:search_end]
        if len(y_region) < 3 or x_region_raw.size != len(y_region):
            continue

        ### 3a) Smooth the local region before derivative calculations.
        y_smooth = y_region.copy()
        if len(y_region) >= 5:
            win = min(smooth_window_max, len(y_region))
            if win % 2 == 0:
                win -= 1
            if win >= 5:
                y_smooth = savgol_filter(y_region, window_length=win, polyorder=polyorder)

        try:
            if np.all(np.isfinite(x_region_raw)) and (x_region_raw[-1] > x_region_raw[0]):
                x_region = np.linspace(float(x_region_raw[0]), float(x_region_raw[-1]), int(len(x_region_raw)))
                y_for_deriv = np.interp(x_region, x_region_raw, y_smooth)
            else:
                x_region = np.arange(search_end - search_start, dtype=float)
                y_for_deriv = y_smooth
        except Exception:
            x_region = np.arange(search_end - search_start, dtype=float)
            y_for_deriv = y_smooth

        ### 3b) Compute first/second derivatives and smooth d2y.
        dy = np.gradient(y_for_deriv, x_region)
        d2y_raw = np.gradient(dy, x_region)
        d2y = d2y_raw.copy()
        if len(y_region) >= 5:
            win_d = min(smooth_window_max, len(y_region))
            if win_d % 2 == 0:
                win_d -= 1
            if win_d >= 5:
                d2y = savgol_filter(d2y_raw, window_length=win_d, polyorder=polyorder)

        ### 3c) Find second-derivative anchor (maximum upslope acceleration).
        edge_guard = int(max(0, min(2, (len(d2y) - 1) // 2)))
        if len(d2y) - (2 * edge_guard) >= 3:
            d2_core = d2y[edge_guard: len(d2y) - edge_guard]
            foot2_local = int(edge_guard + np.argmax(d2_core))
        else:
            foot2_local = int(np.argmax(d2y))

        ### 3d) Walk backward on dy to find low-slope foot onset.
        dy_seg = dy[: foot2_local + 1]
        if dy_seg.size == 0:
            continue
        dy_max = float(np.max(dy_seg))
        picked_local = int(foot2_local)
        slope_thr = np.nan
        if np.isfinite(dy_max) and dy_max > 0:
            slope_thr = 0.08 * dy_max
            for j in range(int(foot2_local), -1, -1):
                if float(dy[j]) <= float(slope_thr):
                    picked_local = int(j)
                    break
        picked = int(search_start + picked_local)

        ### 3e) Apply morphology guardrails and fallback if needed.
        try:
            trough_y = float(np.min(y_values[prev_peak:peak])) if peak > prev_peak + 1 else float(y_values[prev_peak])
            peak_y = float(y_values[peak])
        except Exception:
            trough_y = float(np.min(y_region))
            peak_y = float(np.max(y_region))
        allowed_y = trough_y + float(foot_max_rel_height) * (peak_y - trough_y)
        needs_fallback = (
            picked < 0
            or picked >= len(y_values)
            or picked >= peak - int(max(1, min_samples_before_peak))
            or float(y_values[picked]) > allowed_y
        )
        if needs_fallback:
            seg_pre = y_values[search_start: search_start + foot2_local + 1]
            if seg_pre.size > 0:
                picked = int(search_start + int(np.argmin(seg_pre)))
            else:
                picked = int(search_start + foot2_local)

        foot_indices.append(picked)
        search_windows.append((search_start, search_end))
        debug_rows.append(
            {
                "search_start": int(search_start),
                "search_end": int(search_end),
                "foot2_global": int(search_start + foot2_local),
                "picked_global": int(picked),
                "dy": np.asarray(dy, dtype=float),
                "d2y": np.asarray(d2y, dtype=float),
                "slope_thr": float(slope_thr) if np.isfinite(slope_thr) else np.nan,
            }
        )

    foot_indices = np.asarray(foot_indices, dtype=int)
    if len(foot_indices) < 2:
        raise ValueError("Not enough foot points found to calculate mean wave.")
    ### 4) Build foot-to-foot beats and derive PS/ED points.
    y_s = np.asarray(y_values, dtype=float)
    if len(y_values) >= 5:
        y_s = np.convolve(y_values, np.ones(5) / 5.0, mode="same")
    ps_indices = []
    ed_indices = []
    for i in range(len(foot_indices) - 1):
        a = int(foot_indices[i])
        b = int(foot_indices[i + 1])
        if b <= a + 2:
            continue
        seg = y_s[a:b]
        anchor_in_beat = peak_indices[(peak_indices >= a) & (peak_indices < b)]
        ps_idx = None
        if anchor_in_beat.size > 0:
            aa = anchor_in_beat[np.argmax(y_s[anchor_in_beat])]
            ps_idx = int(aa)
            ps_indices.append(ps_idx)
        else:
            ps_idx = int(a + int(np.argmax(seg)))
            ps_indices.append(ps_idx)

        # ED is constrained to occur after PS within the same beat.
        ed_start = int(max(a, ps_idx + 1))
        if ed_start < b:
            seg_ed = y_s[ed_start:b]
            if seg_ed.size > 0:
                ed_indices.append(int(ed_start + int(np.argmin(seg_ed))))
                continue
        # Fallback for very short post-PS segments.
        ed_indices.append(int(a + int(np.argmin(seg))))
    segment_indices = foot_indices

    ### 5) Segment, align, average, filter outliers, and recompute mean wave.
    interpolated_waves = []
    if verbose:
        _plot_detection_diagnostics(
            x_values=x_values,
            y_values=y_values,
            peak_indices=peak_indices,
            foot_indices=foot_indices,
            ps_indices=ps_indices,
            ed_indices=ed_indices,
            search_windows=search_windows,
            debug_rows=debug_rows,
            smooth_window_max=smooth_window_max,
            polyorder=polyorder,
        )

    for i in range(len(segment_indices) - 1):
        # Extract data for the current segment
        start_index = segment_indices[i]
        end_index = segment_indices[i + 1]
        x_segment = x_values[start_index:end_index]
        y_segment = y_values[start_index:end_index]

        # Shift x-coordinates for alignment (except the first wave)
        if i > 0:
            x_segment = x_segment - (x_segment[0] - x_values[segment_indices[0]])

        # Initialize the common x-axis using the first segment
        if i == 0:
            x_min = x_segment[0]
            x_max = x_segment[-1]
            x_common_points = len(x_segment)
            x_common = np.linspace(x_min, x_max, x_common_points)

        # Interpolate to the common x-axis
        interp_y = interp1d(x_segment, y_segment, kind='linear', fill_value="extrapolate")(x_common)
        interpolated_waves.append(interp_y)

    # Convert the list of interpolated waves to a NumPy array for calculations
    interpolated_waves_np = np.vstack(interpolated_waves)
    total_beats = interpolated_waves_np.shape[0]

    # Calculate the initial average and standard deviation
    average_wave = np.mean(interpolated_waves_np, axis=0)
    std_wave = np.std(interpolated_waves_np, axis=0)
    amplitude_of_ave = np.max(average_wave) - np.min(average_wave)

    if verbose:
        _plot_wave_set(x_common, interpolated_waves, average_wave, std_wave)

    # Outlier rejection: keep beats with most samples within mean ± 20% of mean amplitude.
    threshold_percentage = 80

    filtered_waves = []
    excluded_waves = []
    count_excluded = 0
    for wave in interpolated_waves_np:
        within_range = (wave >= (average_wave - 0.2 * amplitude_of_ave)) & (
            wave <= (average_wave + 0.2 * amplitude_of_ave)
        )
        percentage_within_range = np.sum(within_range) / len(wave) * 100
        if percentage_within_range >= threshold_percentage:
            filtered_waves.append(wave)
        else:
            count_excluded += 1
            excluded_waves.append(wave)
    if verbose:
        print("Waves filtered, num excluded", count_excluded)

    if verbose:
        print(
            f"{len(filtered_waves)} of {len(interpolated_waves)} beats retained "
            f"(>={threshold_percentage}% samples within ±20% amplitude band)."
        )
    # Mean waveform from retained beats only.
    if filtered_waves:
        valid_beats = np.vstack(filtered_waves)
        new_average_wave = np.mean(valid_beats, axis=0)
        new_std_wave = np.std(valid_beats, axis=0)
    else:
        valid_beats = np.empty((0, len(x_common)), dtype=float)
        new_average_wave = average_wave
        new_std_wave = std_wave
    if verbose:
        _plot_average_wave(x_common, new_average_wave)

    num_filtered_waves = valid_beats.shape[0]
    num_excluded_waves = len(excluded_waves)

    diagnostics = {
        "total_beats": total_beats,
        "num_beats_retained": num_filtered_waves,
        "num_beats_excluded": num_excluded_waves,
    }

    return new_average_wave, x_common, valid_beats, diagnostics


def _beat_ps_ed_mean(beat, late_diastolic_frac=0.6):
    """
    PSV, EDV, TAMV and sample indices for one aligned beat (one cardiac cycle).

    PSV is the segment maximum. EDV is the minimum in the late diastolic limb after
    PSV; ``late_diastolic_frac`` skips the early post-systolic region before that search.
    TAMV is the mean over the beat (uniform time sampling assumed).

    :param beat: 1D velocity envelope for one cycle.
    :param late_diastolic_frac: fraction of the post-PSV interval to skip before EDV search.
    :return: (ps, ed, mean_v, ps_idx, ed_idx)
    """
    beat = np.asarray(beat, dtype=float)
    ps_idx = int(np.argmax(beat))
    ps = float(beat[ps_idx])
    mean_v = float(np.mean(beat))

    post_len = beat.size - (ps_idx + 1)
    if post_len > 0:
        late_start = ps_idx + 1 + int(late_diastolic_frac * post_len)
        late_start = min(late_start, beat.size - 1)
        ed_rel = int(np.argmin(beat[late_start:]))
        ed_idx = late_start + ed_rel
        ed = float(beat[ed_idx])
    else:
        ed_idx = int(np.argmin(beat))
        ed = float(beat[ed_idx])

    return ps, ed, mean_v, ps_idx, ed_idx


def _validate_valid_beats(valid_beats):
    """Require a non-empty 2D valid_beats array."""
    valid_beats = np.asarray(valid_beats, dtype=float)
    if valid_beats.ndim != 2 or valid_beats.shape[0] == 0:
        raise ValueError("valid_beats must be a non-empty 2D array (n_beats, n_samples).")
    return valid_beats


def _mean_per_beat(per_beat, index_name):
    """Mean of finite per-beat values."""
    finite = per_beat[np.isfinite(per_beat)]
    if finite.size == 0:
        raise ValueError(f"No beats yielded a finite {index_name}.")
    return float(np.mean(finite)), int(finite.size)


def compute_psv_edv(valid_beats, late_diastolic_frac=0.6):
    """
    PSV, EDV and TAMV for each row of ``valid_beats`` (``mean_wave`` output).

    Uses the same picks as :func:`compute_pi` and :func:`compute_ri`.
    ``psv``, ``edv`` and ``tamv`` are the mean of the per-beat values.

    :param valid_beats: shape (n_beats, n_samples).
    :param late_diastolic_frac: passed to :func:`_beat_ps_ed_mean`.
    :return: dict with per-beat arrays, mean velocities, ``n_beats_used``,
        and index arrays for plotting.
    """
    valid_beats = _validate_valid_beats(valid_beats)
    n = valid_beats.shape[0]
    psv = np.empty(n, dtype=float)
    edv = np.empty(n, dtype=float)
    tamv = np.empty(n, dtype=float)
    ps_idx = np.empty(n, dtype=int)
    ed_idx = np.empty(n, dtype=int)
    for i, beat in enumerate(valid_beats):
        ps, ed, mean_v, p_idx, e_idx = _beat_ps_ed_mean(beat, late_diastolic_frac)
        psv[i] = ps
        edv[i] = ed
        tamv[i] = mean_v
        ps_idx[i] = p_idx
        ed_idx[i] = e_idx

    psv_agg, n_used = _mean_per_beat(psv, "PSV")
    edv_agg, _ = _mean_per_beat(edv, "EDV")
    tamv_agg, _ = _mean_per_beat(tamv, "TAMV")

    return {
        "psv": psv_agg,
        "edv": edv_agg,
        "tamv": tamv_agg,
        "psv_per_beat": psv,
        "edv_per_beat": edv,
        "tamv_per_beat": tamv,
        "ps_idx_per_beat": ps_idx,
        "ed_idx_per_beat": ed_idx,
        "n_beats_used": n_used,
    }


def compute_pi(valid_beats):
    """
    Pulsatility index from rows of ``valid_beats`` (``mean_wave`` output).

    Per beat: PI = (PSV - EDV) / TAMV. ``pi`` is the mean of per-beat values.

    :param valid_beats: shape (n_beats, n_samples).
    :return: dict with keys ``pi``, ``pi_per_beat``, ``n_beats_used``.
    """
    valid_beats = _validate_valid_beats(valid_beats)
    pi_per_beat = np.empty(valid_beats.shape[0], dtype=float)
    for i, beat in enumerate(valid_beats):
        ps, ed, mean_v, _, _ = _beat_ps_ed_mean(beat)
        pi_per_beat[i] = (ps - ed) / mean_v if mean_v > 0 else np.nan

    pi, n_used = _mean_per_beat(pi_per_beat, "PI")
    return {"pi": pi, "pi_per_beat": pi_per_beat, "n_beats_used": n_used}


def compute_ri(valid_beats):
    """
    Resistance index from rows of ``valid_beats`` (``mean_wave`` output).

    Per beat: RI = (PSV - EDV) / PSV. ``ri`` is the mean of per-beat values.

    :param valid_beats: shape (n_beats, n_samples).
    :return: dict with keys ``ri``, ``ri_per_beat``, ``n_beats_used``.
    """
    valid_beats = _validate_valid_beats(valid_beats)
    ri_per_beat = np.empty(valid_beats.shape[0], dtype=float)
    for i, beat in enumerate(valid_beats):
        ps, ed, _mean_v, _, _ = _beat_ps_ed_mean(beat)
        ri_per_beat[i] = (ps - ed) / ps if ps > 0 else np.nan

    ri, n_used = _mean_per_beat(ri_per_beat, "RI")
    return {"ri": ri, "ri_per_beat": ri_per_beat, "n_beats_used": n_used}


def _plot_detection_diagnostics(
    x_values,
    y_values,
    peak_indices,
    foot_indices,
    ps_indices,
    ed_indices,
    search_windows,
    debug_rows,
    smooth_window_max,
    polyorder,
):
    """
    Plot diagnostic panels for beat-foot detection and derivative-based anchors.

    :param x_values: 1D numpy array of x-axis values for the waveform.
    :param y_values: 1D numpy array of waveform amplitudes.
    :param peak_indices: 1D integer array of anchor peak indices.
    :param foot_indices: 1D integer array of detected foot indices.
    :param ps_indices: List/array of peak systolic (PS) indices per beat.
    :param ed_indices: List/array of end-diastolic (ED) indices per beat.
    :param search_windows: List of (start, end) index tuples used as foot search regions.
    :param debug_rows: List of dictionaries containing per-beat debug values
    (search bounds, selected indices, and slope thresholds).
    :param smooth_window_max: Integer max odd window size used for Savitzky-Golay smoothing.
    :param polyorder: Polynomial order for Savitzky-Golay smoothing.
    :return: None. Shows matplotlib figures for interactive diagnostics.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    y_smooth_global = y_values.copy()
    if len(y_values) >= 5:
        win_g = min(smooth_window_max, len(y_values))
        if win_g % 2 == 0:
            win_g -= 1
        if win_g >= 5:
            y_smooth_global = savgol_filter(
                y_values, window_length=win_g, polyorder=polyorder
            )

    ax1.plot(x_values, y_values, label="Waveform", color="blue")
    ax1.plot(
        x_values,
        y_smooth_global,
        label="Smoothed (for derivatives)",
        color="C1",
        linewidth=1.2,
        alpha=0.9,
    )
    ax1.scatter(
        x_values[peak_indices],
        y_values[peak_indices],
        color="black",
        label="Peaks",
        zorder=5,
    )
    ax1.scatter(
        x_values[foot_indices],
        y_values[foot_indices],
        color="red",
        label="Detected feet",
        zorder=6,
    )
    if len(ps_indices) > 0:
        ax1.scatter(
            x_values[np.asarray(ps_indices, dtype=int)],
            y_values[np.asarray(ps_indices, dtype=int)],
            color="C2",
            s=26,
            marker="^",
            label="PS (foot-beat)",
            zorder=7,
        )
    if len(ed_indices) > 0:
        ax1.scatter(
            x_values[np.asarray(ed_indices, dtype=int)],
            y_values[np.asarray(ed_indices, dtype=int)],
            color="C3",
            s=26,
            marker="v",
            label="ED (foot-beat)",
            zorder=7,
        )
    for k, (s, e) in enumerate(search_windows):
        ax1.axvspan(
            x_values[s],
            x_values[e - 1],
            color="orange",
            alpha=0.12,
            label="Search window" if k == 0 else None,
        )
    if foot_indices.size > 0:
        fp = np.asarray(foot_indices, dtype=int)
        fp = fp[(fp >= 0) & (fp < len(x_values))]
        if fp.size > 0:
            if int(fp[0]) > 0:
                ax1.axvspan(
                    x_values[0],
                    x_values[int(fp[0])],
                    color="#ff6b6b",
                    alpha=0.08,
                    zorder=0,
                    label="Incomplete region",
                )
            ap = np.asarray(peak_indices, dtype=int)
            ap = ap[(ap >= 0) & (ap < len(x_values))]
            if ap.size > 0 and int(ap[-1]) > int(fp[-1]):
                ax1.axvspan(
                    x_values[int(fp[-1])],
                    x_values[-1],
                    color="#ff6b6b",
                    alpha=0.08,
                    zorder=0,
                    label=None,
                )
    ax1.set_title("Waveform with anchor peaks, feet, PS and ED")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    dy_g = np.gradient(y_smooth_global, x_values)
    d2y_g = np.gradient(dy_g, x_values)
    ax2.axhline(0.0, color="0.7", linewidth=1)
    ax2.plot(
        x_values,
        dy_g,
        linestyle="-",
        linewidth=1.2,
        label="First derivative (dy/dx) of smoothed signal",
    )
    for k, (s, e) in enumerate(search_windows):
        ax2.axvspan(
            x_values[s],
            x_values[e - 1],
            color="orange",
            alpha=0.12,
            label="Search window" if k == 0 else None,
        )

    for k, row in enumerate(debug_rows):
        s = int(row["search_start"])
        e = int(row["search_end"])
        ub = int(row["foot2_global"])
        picked = int(row["picked_global"])
        slope_thr = float(row["slope_thr"]) if np.isfinite(row["slope_thr"]) else np.nan
        if s < 0 or e <= s or e > len(x_values):
            continue
        if np.isfinite(slope_thr):
            ax2.plot(
                [x_values[s], x_values[e - 1]],
                [slope_thr, slope_thr],
                ":",
                color="gray",
                alpha=0.7,
                label="dy threshold" if k == 0 else None,
            )
        if 0 <= ub < len(x_values):
            ax2.scatter(
                [x_values[ub]],
                [dy_g[ub]],
                marker="s",
                facecolors="none",
                edgecolors="black",
                s=24,
                label="Second-derivative anchor" if k == 0 else None,
                zorder=6,
            )
        if 0 <= picked < len(x_values):
            ax2.scatter(
                [x_values[picked]],
                [dy_g[picked]],
                marker="D",
                color="red",
                s=22,
                label="Selected foot (low-slope onset)" if k == 0 else None,
                zorder=7,
            )

    ax2.set_title("First derivative with second-derivative anchors and selected feet")
    ax2.set_ylabel("First derivative (dy/dx)")
    ax2.legend()
    ax2.grid(True)

    ax3.axhline(0.0, color="0.7", linewidth=1)
    ax3.plot(
        x_values,
        d2y_g,
        linestyle="-",
        linewidth=1.2,
        label="Second derivative (d²y/dx²) of smoothed signal",
    )
    for k, (s, e) in enumerate(search_windows):
        ax3.axvspan(
            x_values[s],
            x_values[e - 1],
            color="orange",
            alpha=0.12,
            label="Search window" if k == 0 else None,
        )

    for k, row in enumerate(debug_rows):
        ub = int(row["foot2_global"])
        if 0 <= ub < len(x_values):
            ax3.scatter(
                [x_values[ub]],
                [d2y_g[ub]],
                marker="s",
                facecolors="none",
                edgecolors="black",
                s=24,
                label="Second-derivative anchor" if k == 0 else None,
                zorder=6,
            )

    ax3.set_title("Second derivative with anchor points")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Second derivative (d²y/dx²)")
    ax3.legend()
    ax3.grid(True)

    fig.tight_layout()
    plt.show()


def _plot_wave_set(x_common, interpolated_waves, average_wave, std_wave):
    """
    Plot aligned beat waveforms with their mean and +/- 1 standard deviation.

    :param x_common: 1D numpy array of common x-axis points used after interpolation.
    :param interpolated_waves: Iterable of 1D arrays representing aligned beat waveforms.
    :param average_wave: 1D numpy array containing the pointwise mean waveform.
    :param std_wave: 1D numpy array containing the pointwise waveform standard deviation.
    :return: None. Shows a matplotlib figure.
    """
    plt.figure(figsize=(10, 6))
    plt.title("Set of Waveforms")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    for wave_index, waveform in enumerate(interpolated_waves):
        plt.plot(x_common, waveform, label=f"Waveform {wave_index}")
    plt.plot(x_common, average_wave, label='Average wave', linestyle='-.')
    plt.plot(x_common, average_wave + std_wave, label='Average wave + SD', linestyle='--')
    plt.plot(x_common, average_wave - std_wave, label='Average wave - SD', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_average_wave(x_common, new_average_wave):
    """
    Plot the final averaged waveform after outlier-beat filtering.

    :param x_common: 1D numpy array of common x-axis points.
    :param new_average_wave: 1D numpy array of final averaged waveform values.
    :return: None. Shows a matplotlib figure.
    """
    plt.figure(figsize=(10, 6))
    plt.title("Average waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.plot(x_common, new_average_wave)
    plt.grid(True)
    plt.show()


class MetaData:
    def __init__(self, dataframe=None, database_path=None):
        self.dataframe = copy.deepcopy(dataframe)
        if (dataframe is not None) ^ bool(database_path):
            if database_path:
                self.dataframe = copy.deepcopy(self.__parse_metadata_db(database_path))

            assert self.dataframe.index.is_unique, (
                "Dataframe indexes must be unique, there is a risk using the truncated hash "
                "values but the risk is small")
        else:
            print("INITIALISATION ERROR, user should provide ONE of <dataframe> or <path to a sql database>\n"
                  "NOT both or neither")

    @classmethod
    def from_dataframe(cls, dataframe):
        return cls(dataframe=dataframe)

    @classmethod
    def from_db_path(cls, db_path):
        return cls(database_path=db_path)

    def __process_row(self, row):
        row = list(row)
        row[0] = int(row[0].split('-')[-1])  # truncate Daphne-<number> to just <number>
        row[5] = row[5][:8]  # truncate the image hash to the first 8 characters
        return row

    def __parse_metadata_db(self, sql_database):
        assert sql_database.exists(), "Invalid database provided, it does not exist"
        # read db into RAM as pandas dataframe
        con = sqlite3.connect(sql_database)
        cur = con.cursor()
        res = cur.execute("PRAGMA table_info(daphne_metadata)")
        cols = [x[1] for x in res.fetchall()]
        col_str = ','.join(cols)
        res = cur.execute(f"SELECT {col_str} FROM daphne_metadata")
        metadata = res.fetchall()
        con.close()

        cols[0] = 'Daphne Number'
        metadata = [self.__process_row(x) for x in metadata]
        dataframe = pd.DataFrame(metadata, columns=cols).set_index('image_hash')
        pd.options.display.max_columns = len(cols)
        dataframe[cols[7:]] = dataframe[cols[7:]].apply(pd.to_numeric, errors='coerce')
        dataframe['HR'] = dataframe['HR'].astype(int)
        dataframe['datetime'] = pd.to_datetime(dataframe['dateofexam'],
                                                                 format='%Y%m%d')
        # drop columns with no unique data
        for col in dataframe.columns:
            if dataframe[col].unique().size == 1:
                print(f"Dropping {col} as all rows have the value: {dataframe[col].unique()[0]}")
                dataframe.drop(col, axis=1, inplace=True)

        return dataframe

    def __get__column(self, col):
        return self.dataframe[col]

    def summary_data(self):
        print(self.dataframe.describe(include='all'))

    def mean(self, column):
        print(self.__get__column(column).mean())

    def Get_Fields(self, cols):
        return MetaData.from_dataframe(self.dataframe[cols])

    def Get_Field_Names(self):
        fields = self.dataframe.columns.to_list()
        return fields

    def Get_Directories(self):
        dirs = {}
        for daphne_number in self.dataframe['Daphne Number'].unique():
            image_dir = f"Primary/leap/doppler/Daphne-{daphne_number}"
            waveform_dir = f"Derivative/leap/doppler/Daphne-{daphne_number}"
            dirs[daphne_number] = {'image directory': image_dir, "waveform directory": waveform_dir}
        return dirs

    def Get_Daphnes(self, daphnes):
        return MetaData.from_dataframe(self.dataframe[self.dataframe['Daphne Number'].isin(daphnes)])

    def __get__uniques(self, series):
        return series.unique()

    def Get_Vessels(self):
        series = self.__get__column('prefix')
        return self.__get__uniques(series).tolist()

    def Get_Index_as_array(self):
        return self.dataframe.index.to_numpy()

    def Get_Num_Entries(self):
        return self.dataframe.shape[0]
    def Query(self, query):
        try:
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            caller_globals = caller_frame.f_globals

            subset = self.dataframe.query(query, local_dict = caller_locals, global_dict=caller_globals)
            return MetaData.from_dataframe(subset)
        except:
            print("Error executing your query, sql like queries are accepted e.g:\n"
                  "`Daphne Number` == 44 and HR < 120 or\n"
                  "PI > 1"
                  "prefix == \"LT\""
                  "You can refer to variables in the environment by prefixing them with an ‘@’ character like @a + b."
                  "You can refer to column names that are not valid Python variable names by surrounding them in backticks.")
        return
    def Get_Earliest_Scan_Datetime(self):
        return self.dataframe['datetime'].min()

    def Get_Latest_Scan_Datetime(self):
        return self.dataframe['datetime'].max()

    def Sort(self, field, ascending=True):
        """Sort the database by the values in the <field>, ascending by default controlled by the <ascending> parameter,
        this happens in place"""
        self.dataframe.sort_values(field, axis='columns', ascending=ascending)
    def Summary(self, verbose=False):
        num_images = self.Get_Num_Entries()
        fields = self.Get_Field_Names()
        if 'Daphne Number' in fields:
            unique_patients = self.dataframe['Daphne Number'].unique().size
            print(f"This data containts information from {unique_patients} distinct patient(s)")
        num_fields = len(fields)
        print(
            f"This data contains information from {num_images} unique images, each with {num_fields} fields, {fields}")
        if 'prefix' in fields:
            vessels = self.Get_Vessels()
            if 'N/A' in vessels:
                unknown_index = vessels.index('N/A')
                print(
                    f"There are {len(vessels) - 1} distinct vessesl that are imaged in this dataset, {[x for x in vessels if x != 'N/A']},including some that have "
                    f"not been specified, these are labelled {vessels[unknown_index]}")
            else:
                print(f"There are {len(vessels)} distinct vessesl, {vessels} that are imaged in this dataset")
        if verbose:
            self.summary_data()

    def Get_Image_Data(self, im_hash):
        """"accepts either singular argumebts specifying a single image index, or a list of image indices"""
        return self.dataframe.loc[im_hash]

    def Generate_timeDelta(self, delta_key, delta_value):
        if delta_key in ['day', 'days', 'month', 'months', 'year', 'years']:
            return pd.Timedelta(delta_value, unit=delta_key)
        else:
            print("USEAGE ERROR: correct useage is Generate_timeDelta(time_units, time_value\n"
                  "Your units must be one of 'day', 'days', 'month', 'months', 'year', 'years'")

    def to_numpy(self):
        return self.dataframe.to_numpy()