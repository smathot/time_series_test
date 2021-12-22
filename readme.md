# Time Series Test

*A statistical test and plotting function for time-series data in general, and data from cognitive-pupillometry experiments in particular*

Sebastiaan Mathôt (@smathot) <br />
Copyright 2021


## Contents

- [Citation](#citation)
- [About](#about)
- [Usage](#usage)
- [Function reference](#function-reference)
- [License](#license)


## Citation

Mathôt, S., & Vilotijević, A. (in prep). *A Hands-on Guide to Cognitive Pupillometry: from Design to Analysis.*


## About

For a more detailed description, see the manuscript above.

This package provides a function (`find()`) that locates and statistically tests effects in time-series data. It does so by splitting the data in a number of subsets (by default 4). It takes one of the subsets (the *test* set) out of the full dataset, and conducts a linear mixed effects model on each sample of the remaining data (the *training* set). The sample with the highest absolute z value in the training set is used as the sample-to-be-tested for the test set. This procedure is repeated for all subsets of the data, and for all fixed effects in the model. Finally, a single linear mixed effects model is conducted for each fixed effects on the samples that were thus identified.

This packages also provides a function (`plot()`) to visualize time-series data to visually annotate the results of `find()`.


## Usage

We will use data from [Zhou, Lorist, and Mathôt (2021)](https://doi.org/10.1101/2021.11.23.469689). In brief, this is data from a visual-working-memory experiment in which participant memorized one or more colors (set size: 1, 2, 3 or 4) of two different types (color type: proto, nonproto) while pupil size was being recorded during a 3s retention interval.

This dataset contains the following columns:

- `pupil`, which is is our dependent measure. It is a baseline-corrected pupil time series of 300 samples, recorded at 100 Hz
- `subject_nr`, which we will use as a random effect
- `set_size`, which we will use as a fixed effect
- `color_type`, which we will use as a fixed effect

First, load the dataset:



```python
from datamatrix import io
dm = io.readpickle('data/zhou_et_al_2021.pkl')
```



The `plot()` function provides a convenient way to plot pupil size over time as a function of one or two factors, in this case set size and color type:



```python
import time_series_test as tst
from matplotlib import pyplot as plt

tst.plot(dm, dv='pupil', hue_factor='set_size', linestyle_factor='color_type')
plt.savefig('img/signal-plot-1.png')
```



![](https://github.com/smathot/time_series_test/raw/master/img/signal-plot-1.png)

From this plot, we can tell that there appear to be effects in the 1500 to 2000 ms interval. To test this, we could perform a linear mixed effects model on this interval, which corresponds to samples 150 to 200.

The model below uses mean pupil size during the 150 - 200 sample range as dependent measure, set size and color type as fixed effects, and a random by-subject intercept. In the more familiar notation of the R package `lme4`, this corresponds to `mean_pupil ~ set_size * color_type + (1 | subject_nr)`. (To use more complex random-effects structures, you can use the `re_formula` argument to `mixedlm()`.)



```python
from statsmodels.formula.api import mixedlm
from datamatrix import series as srs, NAN

dm.mean_pupil = srs.reduce(dm.pupil[:, 150:200])
dm_valid_data = dm.mean_pupil != NAN
model = mixedlm(formula='mean_pupil ~ set_size * color_type',
                data=dm_valid_data, groups='subject_nr').fit()
print(model.summary())
```

__Output:__
``` .text
                    Mixed Linear Model Regression Results
=============================================================================
Model:                    MixedLM       Dependent Variable:       mean_pupil 
No. Observations:         7300          Method:                   REML       
No. Groups:               30            Scale:                    38610.3390 
Min. group size:          235           Log-Likelihood:           -48952.3998
Max. group size:          248           Converged:                Yes        
Mean group size:          243.3                                              
-----------------------------------------------------------------------------
                              Coef.   Std.Err.   z    P>|z|  [0.025   0.975] 
-----------------------------------------------------------------------------
Intercept                    -144.024   17.438 -8.259 0.000 -178.202 -109.846
color_type[T.proto]           -24.133   11.299 -2.136 0.033  -46.278   -1.987
set_size                       49.979    2.906 17.200 0.000   44.284   55.675
set_size:color_type[T.proto]   10.176    4.120  2.470 0.014    2.101   18.251
subject_nr Var               7217.423    9.882                               
=============================================================================

```



The model summary shows that, assuming an alpha level of .05, there are significant main effects of color type (z = -2.136, p = .033), set size (z = 17.2, p < .001), and a significant color-type by set-size interaction (z = 2.47, p = .014). However, we have selectively analyzed a sample range that we knew, based on a visual inspection of the data, to show these effects. This means that our analysis is circular: we have looked at the data to decide where to look! The `find()` function improves this by splitting the data into training and tests sets, as described under [About](#about), thus breaking the circularity.



```python
results = tst.find(dm,  'pupil ~ set_size * color_type',
                   groups='subject_nr', winlen=5)
```




The return value of `find()` is a `dict`, where keys are effect labels and values are named tuples of the following:

- `model`: a model as returned by `mixedlm().fit()`
- `samples`: a set with the sample indices that were used
- `p`: the p-value from the model
- `z`: the z-value from the model



```python
for effect, (model, samples, p, z) in results.items():
    print('{} was tested at samples {} → z = {:.4f}, p = {:.4}'.format(
          effect, samples, z, p))
```

__Output:__
``` .text
Intercept was tested at samples {95} → z = -13.1098, p = 2.892e-39
color_type[T.proto] was tested at samples {160, 170, 175} → z = -2.0949, p = 0.03618
set_size was tested at samples {185, 210, 195, 255} → z = 16.2437, p = 2.475e-59
set_size:color_type[T.proto] was tested at samples {165, 175} → z = 2.5767, p = 0.009974
```



We can pass the `results` to `plot()` to visualize the results:



```python
tst.plot(dm, dv='pupil', hue_factor='set_size', linestyle_factor='color_type',
         results=results)
plt.savefig('img/signal-plot-2.png')
```



![](https://github.com/smathot/time_series_test/raw/master/img/signal-plot-2.png)


## Function reference

### find()



```python
print(tst.find.__doc__)
```

__Output:__
``` .text
Conducts a single linear mixed effects model to a time series, where the
    to-be-tested samples are determined through a validation-test procedure.
    
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
    samples_fe: bool, optional
        Indicates whether sample indices are included as an additive factor
        to the fixed-effects formula. If all splits yielded the same sample
        index, this is ignored.
    samples_re: bool, optional
        Indicates whether sample indices are included as an additive factor
        to the random-effects formula. If all splits yielded the same sample
        index, this is ignored.
    fit_method: str, list of str, or None, optional
        The fitting method, which is passed as the `method` keyword to
        `mixedlm.fit()`. This can be a label or a list of labels, in which
        case different fitting methods are tried in case of convergence errors.
    **kwargs: dict, optional
        Optional keywords to be passed to `mixedlm()`, such as `groups` and
        `re_formula`.
        
    Returns
    -------
    dict
        A dict where keys are effect labels, and values are named tuples
        of `model`, `samples`, `p`, and `z`.
    
```



### plot()



```python
print(tst.plot.__doc__)
```

__Output:__
``` .text
Visualizes a time series, where the signal is plotted as a function of
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
    
```




## License

`biased_memory_toolbox` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
