# coding=utf-8
"""A statistical test and plotting function for time-series data in general, and data from cognitive-pupillometry experiments in particular. Based on linear mixed effects modeling and crossvalidation.
"""
from datamatrix import DataMatrix, SeriesColumn, convert as cnv, \
    series as srs, operations as ops
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import warnings
import logging
from collections import namedtuple

__version__ = '0.6.0'
TEAL = ['#004d40', '#00796b', '#009688', '#4db6ac', '#b2dfdb']
DEEP_ORANGE = ['#bf360c', '#e64a19', '#ff5722', '#ff8a65', '#ffccbc']
LINESTYLES = ['-', '--', ':']
logger = logging.getLogger('time_series_test')


def find(dm, formula, groups, re_formula=None, winlen=1, split=4,
         split_method='interleaved', samples_fe=True, samples_re=True,
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
    groups: str or list of str
        The groups for the random effects, which should be regular (non-series)
        columns in `dm`.
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
        Optional keywords to be passed to `mixedlm()`, such as `groups` and
        `re_formula`.
        
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
        print('xx')
    dm = _trim_dm(dm, formula, groups, re_formula)
    with warnings.catch_warnings():
        if suppress_convergence_warnings:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
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
                                    fit_kwargs=fit_kwargs, samples_re=samples_re,
                                    **kwargs)


def plot(dm, dv, hue_factor, results=None, linestyle_factor=None, hues=None,
         linestyles=None, alpha_level=.05, annotate_intercept=False,
         annotation_hues=None, annotation_linestyle=':'):
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
        A `results` dict as returned by `find()`.
    linestyle_factor: str, optional
        The name of a regular (non-series) column in `dm` that specifies the
        linestyle of the lines for a two-factor plot.
    hues: list or None, optional
        A list of hues to be used as line colors for the first factor.
    linestyles: list or None, optional
        A list of linestyles to be used for the second factor.
    alpha_level: float, optional
        The alpha level (maximum p value) to be used for annotating effects
        in the plot.
    annotate_intercept: bool, optional
        Specifies whether the intercept should also be annotated along with
        the fixed effects.
    annotation_hues: list or None, optional
        A list of hues to be used as line color for the annotations.
    annotation_linestyle: str, optional
        The linestyle for the annotations.
    """
    if hues is None:
        hues = TEAL
    if annotation_hues is None:
        annotation_hues = DEEP_ORANGE
    if linestyles is None:
        linestyles = LINESTYLES
    # Plot the annotations
    annotation_elements = []
    if results is not None:
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
        plt.gca().add_artist(plt.legend(loc='lower right'))
    hue_legend = [
        Line2D([0], [0], color=hues[i1 % len(hues)], label=f1)
        for i1, f1 in enumerate(dm[hue_factor].unique)
    ]
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
    returned by `find()`.

    Parameters
    ----------
    results: dict
        A `results` dict as returned by `find()`.
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
    for colname in dm.column_names:
        if colname in formula or colname in groups or re_formula is None or \
                colname in re_formula:
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
        lm = _lmer_series(dm[ref_indices], formula, winlen=winlen,
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
    test_dm = dm[:]
    dv = formula.split()[0]
    del test_dm[dv]
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


def _lmer_series(dm, formula, winlen=1, fit_kwargs={}, **kwargs):
    
    col = formula.split()[0]
    depth = dm[col].depth
    rm = None
    for i in range(0, depth, winlen):
        logger.debug('sample {}'.format(i))
        wm = dm[:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wm[col] = srs.reduce(srs.window(wm[col], start=i, end=i+winlen))
        wm = wm[col] != np.nan
        try:
            lm = smf.mixedlm(formula, wm, **kwargs).fit(**fit_kwargs)
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
