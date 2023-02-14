# coding=utf-8
"""A statistical test and plotting function for time-series data in general, 
and data from cognitive-pupillometry experiments in particular. Based on linear
mixed effects modeling and crossvalidation.
"""
from datamatrix import DataMatrix, SeriesColumn, convert as cnv, \
    series as srs, operations as ops
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import logging
import re
from collections import namedtuple

__version__ = '0.10.0'
DEFAULT_HUE_COLORMAP = 'Dark2'
DEFAULT_ANNOTATION_COLORMAP = 'brg'
DEEP_ORANGE = ['#bf360c', '#e64a19', '#ff5722', '#ff8a65', '#ffccbc']
LINESTYLES = ['-', '--', ':']
logger = logging.getLogger('time_series_test')


def lmer_crossvalidation_test(dm, formula, groups, re_formula=None, winlen=1,
        split=4, split_method='interleaved', samples_fe=True, samples_re=True,
        localizer_re=False, fit_method=None,
        suppress_convergence_warnings=False, fit_kwargs=None, **kwargs):
    """Conducts a single linear mixed effects model to a time series, where the
    to-be-tested samples are determined through crossvalidation.
    
    This function uses `mixedlm()` from the `statsmodels` package. See the
    statsmodels documentation for a more detailed explanation of the
    parameters.
    
    Parameters
    ----------
    dm: DataMatrix
        The dataset
    formula: str
        A formula that describes the dependent variable, which should be the
        name of a series column in `dm`, and the fixed effects, which should
        be regular (non-series) columns.
    groups: str or None or list of str
        The groups for the random effects, which should be regular (non-series)
        columns in `dm`. If `None` is specified, then all analyses are based
        on a regular multiple linear regression (instead of linear mixed 
        effects model).
    re_formula: str or None
        A formula that describes the random effects, which should be regular
        (non-series) columns in `dm`.
    winlen: int, optional
        The number of samples that should be analyzed together, i.e. a 
        downsampling window to speed up the analysis.
    split: int, optional
        The number of splits that the analysis should be based on.
    split_method: str, optional
        If 'interleaved', the data is split in a regular interleaved fashion,
        such that the first row goes to the first subset, the second row to the
        second subset, etc. If 'random', the data is split randomly in subsets.
        Interleaved splitting is deterministic (i.e. it results in the same
        outcome each time), but random splitting is not.
    samples_fe: bool, optional
        Indicates whether sample indices are included as an additive factor
        to the fixed-effects formula. If all splits yielded the same sample
        index, this is ignored.
    samples_re: bool, optional
        Indicates whether sample indices are included as an additive factor
        to the random-effects formula. If all splits yielded the same sample
        index, this is ignored.
    localizer_re: bool, optional
        Indicates whether a random effects structure as specified using the
        `re_formula` keyword should also be used for the localizer models,
        or only for the final model.
    fit_kwargs: dict or None, optional
        A `dict` that is passed as keyword arguments to `mixedlm.fit()`. For
        example, to specify the nm as the fitting method, specify
        `fit_kwargs={'fit': 'nm'}`.
    fit_method: str, list of str, or None, optional
        Deprecated. Use `fit_kwargs` instead.
    suppress_convergence_warnings: bool, optional
        Installs a warning filter to suppress conververgence (and other)
        warnings.
    **kwargs: dict, optional
        Optional keywords to be passed to `mixedlm()`.
        
    Returns
    -------
    dict
        A dict where keys are effect labels, and values are named tuples
        of `model`, `samples`, `p`, and `z`.
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    if fit_method is not None:
        warnings.warn(
            'The fit_method keyword is deprecated. Use fit_kwargs instead',
            DeprecationWarning)
        fit_kwargs['method'] = fit_method
    dm = _trim_dm(dm, formula, groups, re_formula)
    with warnings.catch_warnings():
        if suppress_convergence_warnings:
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
        logger.debug('running localizer')
        dm.__lmer_localizer__ = _lmer_run_localizer(
            dm, formula, groups, winlen=winlen, split=split,
            split_method=split_method,
            re_formula=re_formula if localizer_re else None,
            fit_kwargs=fit_kwargs, **kwargs)
        logger.debug('testing localizer results')
        return _lmer_test_localizer(dm, formula, groups, re_formula=re_formula,
                                    winlen=winlen, samples_fe=samples_fe,
                                    fit_kwargs=fit_kwargs,
                                    samples_re=samples_re, **kwargs)


def lmer_series(dm, formula, winlen=1, fit_kwargs={}, **kwargs):
    """Performs a sample-by-sample linear-mixed-effects analysis. See
    `lmer_crossvalidation()` for an explanation of the arguments.
    
    Parameters
    ----------
    dm: DataMatrix
    formula: str
    winlen: int, optional
    fit_kwargs: dict, optional
    **kwargs: dict, optional
    
    Returns
    -------
    DataMatrix
        A DataMatrix with one row per effect, including the intercept, and
        three series columns with the same depth as the dependent measure
        specified in the formula:
        
        - `est`: the slope
        - `p`: the p value
        - `z`: the z value
        - `se`: the standard error
    """
    terms = _terms(formula, **kwargs)
    # For efficient memory use, we first strip all columns except the relevant
    # ones from the datamatrix. We also create an ever leaner copy that doesn't
    # contain the dependent variable, which is a series column. This leaner
    # copy will be filled in with a single (non-series) column for the dv for
    # each iteration of the loop.
    wm = dm[terms]
    wm_no_dv = dm[terms[1:]]
    dv = terms[0]
    depth = dm[dv].depth
    rm = None
    kwargs = kwargs.copy()
    if kwargs.get('groups', None) is None:
        logger.warning('no groups specified, using ordinary least squares')
        if 'groups' in kwargs:
            del kwargs['groups']
        if 're_formula' in kwargs:
            del kwargs['re_formula']
        model = smf.ols
    else:
        model = smf.mixedlm
    for i in range(0, depth, winlen):
        logger.debug('sample {}'.format(i))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wm_no_dv[dv] = srs.reduce(
                srs.window(wm[dv], start=i, end=i+winlen))
        valid_dm = wm_no_dv[dv] != np.nan
        try:
            lm = model(formula, valid_dm, **kwargs).fit(**fit_kwargs)
        except np.linalg.LinAlgError as e:
            warnings.warn('failed to fit mode: {}'.format(e))
            continue
        length = len(lm.model.exog_names)
        if rm is None:
            rm = DataMatrix(length=length)
            rm.effect = lm.model.exog_names
            rm.p = SeriesColumn(depth=depth)
            rm.z = SeriesColumn(depth=depth)
            rm.est = SeriesColumn(depth=depth)
            rm.se = SeriesColumn(depth=depth)
        for sample in range(i, min(depth, i + winlen)):
            rm.p[:, sample] = list(lm.pvalues[:length])
            rm.z[:, sample] = list(lm.tvalues[:length])
            rm.est[:, sample] = list(lm.params[:length])
            rm.se[:, sample] = list(lm.bse[:length])
    return rm


def lmer_permutation_test(dm, formula, groups, re_formula=None, winlen=1,
                          suppress_convergence_warnings=False, fit_kwargs={},
                          iterations=1000, cluster_p_threshold=.05, **kwargs):
    """Performs a cluster-based permutation test based on sample-by-sample
    linear-mixed-effects analyses. The permutation test identifies clusters
    based on p-value threshold and uses the absolute of the summed z-values of
    the clusters as test statistic.
    
    If no clusters reach the threshold, the test is skipped right away. The
    Intercept is ignored for this criterion, because the intercept usually has
    significant clusters that we're not interested in.
    
    *Warning:* This is generally an extremely time-consuming analysis because
    it requires thousands of lmers to be run.
    
    See `lmer_crossvalidation()` for an explanation of the arguments.
    
    Parameters
    ----------
    dm: DataMatrix
    formula: str
    groups: str
    re_formula: str or None, optional
    winlen: int, optional
    suppress_convergence_warnings: bool, optional
    fit_kwargs: dict, optional
    iterations: int, optional
        The number of permutations to run.
    cluster_p_threshold: float or None, optional
        The maximum p-value for a sample to be considered part of a cluster.
    **kwargs: dict, optional
    
    Returns
    -------
    dict
        A dict with effects as keys and lists of clusters defined by
        (start, end, z-sum, hit proportion) tuples. The p-value is
        1 - hit proportion.
    """
    dm = _trim_dm(dm, formula, groups, re_formula)
    terms = _terms(formula, **kwargs)
    dv = terms[0]
    # First conduct a regular sample-by-sample lme, and get the size of the
    # largest clusters for each effect
    with warnings.catch_warnings():
        if suppress_convergence_warnings:
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
        rm_obs = lmer_series(dm, formula=formula, groups=groups,
                             re_formula=re_formula, winlen=winlen,
                             fit_kwargs=fit_kwargs, **kwargs)
    cluster_obs = _clusters(rm_obs, cluster_p_threshold)
    # This is where we will store the hits for each of the observed clusters
    cluster_hits = {effect: [0] * len(clusters)
                    for effect, clusters in cluster_obs.items()}
    logger.info(f'observed clusters: {cluster_obs}')
    # Now run through all iterations
    for i in range(iterations):
        # If there are no significant clusters, we'll skip the test altogether
        # to save time. The Intercept doesn't count here, because the intercept
        # generally does have significant clusters but we're not interested in
        # those.
        if not any(clusters for effect, clusters in cluster_obs.items()
                   if effect != 'Intercept'):
            logger.info(f'no clusters reach threshold, skipping test')
            break
        logger.info(f'start of iteration {i}')
        if groups is None:
            # If no groups are specified (and we are therefore falling back
            # to a normal multiple linear regression), we simply shuffle the
            # full datamatrix.
            for term in terms:
                dm[term] = ops.shuffle(dm[term])
        else:
            # Permute the order of the terms while keeping groups intact
            for group, gdm in ops.split(dm[groups]):
                for term in terms:
                    dm[term][gdm] = ops.shuffle(gdm[term])
        # Conduct a sample-by-sample lme on the permuted data, and get the
        # largest (spurious) clusters
        with warnings.catch_warnings():
            if suppress_convergence_warnings:
                warnings.simplefilter(action='ignore',
                                      category=ConvergenceWarning)
            rm_it = lmer_series(dm, formula=formula, groups=groups,
                                re_formula=re_formula, winlen=winlen,
                                fit_kwargs=fit_kwargs, **kwargs)
        cluster_it = _clusters(rm_it, cluster_p_threshold)
        logger.info(f'observed clusters: {cluster_obs}')
        logger.info(f'permuted clusters: {cluster_it}')
        # For each effect, when the observed cluster size exceeds the spurious
        # cluster size, consider this a hit
        for effect in cluster_hits:
            if not cluster_it[effect]:
                zsum_it = 0
            else:
                # Get the biggers spurious cluster
                _, _, zsum_it = cluster_it[effect][0]
            logger.info(f'permutation zsum for {effect} = {zsum_it:.2f}')
            for j, (_, _, zsum_obs) in enumerate(cluster_obs[effect]):
                if zsum_obs > zsum_it:
                    logger.info(f'hit for {effect} cluster {j} '
                                f'({zsum_obs:.2f} > {zsum_it:.2f})')
                    cluster_hits[effect][j] += 1
                else:
                    logger.info(f'miss for {effect} cluster {j} '
                                f'({zsum_obs:.2f} <= {zsum_it:.2f})')
        logger.info(f'hits after iteration {i}: {cluster_hits}')
    # Return a dict with a list of (start, end, zsum, pvalue) tuples
    cluster_pvalues = {}
    for effect, hitlist in cluster_hits.items():
        cluster_pvalues[effect] = []
        for hits, (start, end, zsum) in zip(hitlist, cluster_obs[effect]):
            cluster_pvalues[effect].append(
                (start, end, zsum, hits / iterations))
    return cluster_pvalues


def plot(dm, dv, hue_factor, results=None, linestyle_factor=None, hues=None,
         linestyles=None, alpha_level=.05, annotate_intercept=False,
         annotation_hues=None, annotation_linestyle=':', legend_kwargs=None,
         annotation_legend_kwargs=None):
    """Visualizes a time series, where the signal is plotted as a function of
    sample number on the x-axis. One fixed effect is indicated by the hue
    (color) of the lines. An optional second fixed effect is indicated by the
    linestyle. If the `results` parameter is used, significant effects are
    annotated in the figure.
    
    Parameters
    ----------
    dm: DataMatrix
        The dataset
    dv: str
        The name of the dependent variable, which should be a series column
        in `dm`.
    hue_factor: str
        The name of a regular (non-series) column in `dm` that specifies the
        hue (color) of the lines.
    results: dict, optional
        A `results` dict as returned by `lmer_crossvalidation()`.
    linestyle_factor: str, optional
        The name of a regular (non-series) column in `dm` that specifies the
        linestyle of the lines for a two-factor plot.
    hues: str, list, or None, optional
        The name of a matplotlib colormap or a list of hues to be used as line
        colors for the hue factor.
    linestyles: list or None, optional
        A list of linestyles to be used for the second factor.
    alpha_level: float, optional
        The alpha level (maximum p value) to be used for annotating effects
        in the plot.
    annotate_intercept: bool, optional
        Specifies whether the intercept should also be annotated along with
        the fixed effects.
    annotation_hues: str, list, or None, optional
        The name of a matplotlib colormap or a list of hues to be used for the
        annotations if `results` is provided.
    annotation_linestyle: str, optional
        The linestyle for the annotations.
    legend_kwargs: None or dict, optional
        Optional keywords to be passed to `plt.legend()` for the factor legend.
    annotation_legend_kwargs: None or dict, optional
        Optional keywords to be passed to `plt.legend()` for the annotation
        legend.
    """
    cols = [dv]
    if hue_factor is not None:
        cols.append(hue_factor)
    if linestyle_factor is not None:
        cols.append(linestyle_factor)
    dm = dm[cols]
    if hues is None:
        hues = DEFAULT_HUE_COLORMAP
    if isinstance(hues, str):
        hues = _colors(hues, dm[hue_factor].count)
    if linestyles is None:
        linestyles = LINESTYLES
    # Plot the annotations
    annotation_elements = []
    if results is not None:
        if annotation_hues is None:
            annotation_hues = DEFAULT_ANNOTATION_COLORMAP
        if isinstance(annotation_hues, str):
            annotation_hues = _colors(annotation_hues, len(results))
        i = 0
        for effect, result in results.items():
            if effect == 'Intercept' and not annotate_intercept:
                continue
            if result.p >= alpha_level:
                continue
            hue = annotation_hues[i % len(annotation_hues)]
            annotation_elements.append(
                plt.axvline(np.mean(list(result.samples)),
                            linestyle=annotation_linestyle,
                            color=hue,
                            label='{}: p = {:.4f}'.format(effect, result.p)))
            i += 1
    # Adjust x axis
    plt.xlim(0, dm[dv].depth)
    # Plot the traces
    x = np.arange(0, dm[dv].depth)
    for i1, (f1, dm1) in enumerate(ops.split(dm[hue_factor])):
        hue = hues[i1 % len(hues)]
        if linestyle_factor is None:
            n = (~np.isnan(dm1[dv])).sum(axis=0)
            y = dm1[dv].mean
            yerr = dm1[dv].std / np.sqrt(n)
            ymin = y - yerr
            ymax = y + yerr
            plt.fill_between(x, ymin, ymax, color=hue, alpha=.2)
            plt.plot(y, color=hue, linestyle=linestyles[0])
        else:
            for i2, (f2, dm2) in enumerate(ops.split(dm1[linestyle_factor])):
                linestyle = linestyles[i2 % len(linestyles)]
                n = (~np.isnan(dm2[dv])).sum(axis=0)
                y = dm2[dv].mean
                yerr = dm2[dv].std / np.sqrt(n)
                ymin = y - yerr
                ymax = y + yerr
                plt.fill_between(x, ymin, ymax, color=hue, alpha=.2)
                plt.plot(y, color=hue, linestyle=linestyle)
    # Implement legend
    if annotation_elements:
        if annotation_legend_kwargs is not None:
            annotation_legend = plt.legend(**annotation_legend_kwargs)
        else:
            annotation_legend = plt.legend(loc='lower right')
        plt.gca().add_artist(annotation_legend)
    hue_legend = [
        Line2D([0], [0], color=hues[i1 % len(hues)], label=f1)
        for i1, f1 in enumerate(dm[hue_factor].unique)
    ]
    if legend_kwargs is not None:
        legend = plt.gca().legend(handles=hue_legend, **legend_kwargs)
    else:
        legend = plt.gca().legend(
            handles=hue_legend,
            title=hue_factor,
            loc='upper left')
    if linestyle_factor is not None:
        plt.gca().add_artist(legend)
        linestyle_legend = [
            Line2D([0], [0], color='black',
                   linestyle=linestyles[i2 % len(linestyles)], label=f2)
            for i2, f2 in enumerate(dm[linestyle_factor].unique)
        ]
        plt.gca().legend(handles=linestyle_legend, title=linestyle_factor,
                         loc='upper right')


def summarize(results, detailed=False):
    """Generates a string with a human-readable summary of a results `dict` as
    returned by `lmer_crossvalidation()`.

    Parameters
    ----------
    results: dict
        A `results` dict as returned by `lmer_crossvalidation()`.
    detailed: bool, optional
        Indicates whether model details should be included in the summary.

    Returns
    -------
    str
    """
    summary = []
    for effect, (model, samples, p, z) in results.items():
        summary.append(
            '{} was tested at samples {} → z = {:.4f}, p = {:.4}'.format(
               effect, samples, z, p))
        if detailed:
            summary += ['', str(model.summary())]
    return '\n'.join(summary)


def _trim_dm(dm, formula, groups, re_formula):
    """Removes unnecessary columns from the datamatrix"""
    trimmed_dm = DataMatrix(length=len(dm))
    if groups is None:
        groups = []
    if re_formula is None:
        re_formula = []
    for colname in dm.column_names:
        if colname in formula or colname in groups or colname in re_formula:
            logger.debug('keeping column {}'.format(colname))
            trimmed_dm[colname] = dm[colname]
        else:
            logger.debug('trimming column {}'.format(colname))
    return trimmed_dm


def _interleaved_indices(length, split):
    
    split_indices = []
    for start in range(split):
        test_indices = [i for i in range(start, length, split)]
        ref_indices = [i for i in range(length) if i not in test_indices]
        split_indices.append((test_indices, ref_indices))
    return split_indices


def _random_indices(length, split):
    
    indices = np.arange(length)
    np.random.shuffle(indices)
    split_indices = []
    chunk_size = length // split
    for start in range(split):
        split_indices.append((indices[:chunk_size], indices[chunk_size:]))
        indices = np.roll(indices, chunk_size)
    return split_indices


def _lmer_run_localizer(dm, formula, groups, re_formula=None, winlen=1,
                        split=4, split_method='interleaved', fit_kwargs={},
                        **kwargs):
    
    if split_method == 'interleaved':
        split_indices = _interleaved_indices(len(dm), split)
    elif split_method == 'random':
        split_indices = _random_indices(len(dm), split)
    else:
        raise ValueError('invalid split_method: {}'.format(split_method))
    # Loop through all test and ref indices, get the corresponding datamatrix
    # objects, and run an lmer on the reference matrix and use this as the
    # localizer for the test matrix.
    result_dm = None
    for test_indices, ref_indices in split_indices:
        logger.debug('test size: {}, reference size: {}'.format(
            len(test_indices), len(ref_indices)))
        lm = lmer_series(dm[ref_indices], formula, winlen=winlen,
                         groups=groups, re_formula=re_formula,
                         fit_kwargs=fit_kwargs, **kwargs)
        if result_dm is None:
            result_dm = dm[tuple()]
            result_dm.lmer_localize = SeriesColumn(depth=len(lm))
        best_sample = np.argmax(np.abs(lm.z), axis=1)
        result_dm.lmer_localize[test_indices, :] = best_sample
        logger.debug('best sample: {}'.format(best_sample))
    return result_dm.lmer_localize


def _lmer_test_localizer(dm, formula, groups, re_formula=None, winlen=1,
                         target_col='__lmer_localizer__', samples_fe=False,
                         samples_re=False, fit_kwargs={}):
    terms = _terms(formula, groups=groups, re_formula=re_formula)
    test_dm = dm[terms[1:]]
    dv = terms[0]
    signal = dm[dv]._seq
    indices = np.array(dm[target_col]._seq, dtype=int)
    results = {}
    Results = namedtuple('LmerTestLocalizerResults', 
                         ['model', 'samples', 'p', 'z'])
    for effect in range(indices.shape[1]):
        mean_signal = np.empty(indices.shape[0])
        samples = np.empty(indices.shape[0])
        for row in range(indices.shape[0]):
            # The indices can be two-dimensional, in which case separate
            # indices are specified for each effect, or one-dimensional, in
            # which case the same index is used for all effects.
            if len(indices.shape) == 2:
                index = indices[row, effect]
            else:
                index = indices[row]
            mean_signal[row] = np.nanmean(signal[row, index:index + winlen])
            samples[row] = index
        test_dm[dv] = mean_signal
        test_dm.__lmer_samples__ = samples
        _formula = formula
        _re_formula = re_formula
        if test_dm.__lmer_samples__.count > 1:
            test_dm.__lmer_samples__ = ops.z(test_dm.__lmer_samples__)
            if samples_fe:
                _formula += ' + __lmer_samples__'
            if samples_re and re_formula is not None:
                _re_formula += ' + __lmer_samples__'
        lm = smf.mixedlm(_formula, test_dm[dv] != np.nan, groups=groups,
                         re_formula=_re_formula).fit(**fit_kwargs)
        effect_name = lm.model.exog_names[effect]
        results[effect_name] = Results(model=lm,
                                       samples=set(indices[:, effect]),
                                       p=lm.pvalues[effect],
                                       z=lm.tvalues[effect])
    return results


def _split_terms(formula):
    """Extracts all terms from a formula."""
    return [term for term in re.split('[ ~+%*]', formula) if term.strip()]
    

def _terms(formula, **kwargs):
    """Extracts all terms from a formula, including those specified in groups
    and re_formula, which are optionally specified through **kwargs
    """
    terms = _split_terms(formula)
    if kwargs.get('groups', None) is not None:
        terms.append(kwargs['groups'])
    if kwargs.get('re_formula', None) is not None:
        terms += _split_terms(kwargs['re_formula'])
    return terms


def _colors(colormap, n):
    cm = plt.colormaps[colormap]
    return [cm(int(hue)) for hue in np.linspace(0, cm.N, n)]


def _clusters(rm, cluster_p_threshold):
    """Extracts the largest clusters based on a p-threshold of .05 based on a
    datamatrix as returned by lmer_series(). Returns a dict with effects as
    keys and a list of (start, end, zsum) tuples as values. The list is sorted
    by the zsum.
    """
    clusters = {}
    for row in rm:
        pi = None
        clusters[row.effect] = []
        # Loop through all indices that are part of clusters
        for i in np.where(row.p < cluster_p_threshold)[0]:
            # We're entering a new cluster
            if i - 1 != pi:
                # Store previous cluster
                if pi is not None:
                    clusters[row.effect].append((start, i, abs(z_sum)))
                start = i
                z_sum = 0
            z_sum += row.z[i]
            pi = i
        if pi is not None:
            # Store last cluster
            clusters[row.effect].append((start, i, abs(z_sum)))
        clusters[row.effect].sort(key=lambda i: -i[2])
    return clusters


# alias for backwards compatibility
find = lmer_crossvalidation_test
