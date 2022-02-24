# Time Series Test

*A statistical test and plotting function for time-series data in general, and data from cognitive-pupillometry experiments in particular. Based on linear mixed effects modeling and crossvalidation.*

Sebastiaan Mathôt (@smathot) <br />
Copyright 2021 - 2022


## Contents

- [Citation](#citation)
- [About](#about)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Function reference](#function-reference)
- [License](#license)


## Citation

Mathôt, S., & Vilotijević, A. (in prep). *A Hands-on Guide to Cognitive Pupillometry: from Design to Analysis.*

*This is a work in progress. A preprint of this manuscript will be made available soon.*


## About

In general terms, this package implements a statistical test for a specific-yet-common question when analyzing time-series data:

> Do one or more independent variables affect a continuously recorded dependent variable (a 'time series') at any point in time?

When to use this test:

- For time series consisting of only a single component, that is, when each independent variable has only a single effect on the time series. An example of this is the effect of stimulus intensity on pupil size, when presenting light flashes of different intensities.
- When you do not know a priori which time points to test.

When *not* to use this test:

- For time series that contain multiple components, that is, when each independent variable affects the time series in multiple ways that change over time. An example of this is the effect of visual attention on lateralized EEG recordings, where different EEG components emerge at different points in time.
- When you know a priori which time points to test.

More specifically, this package provides a function (`find()`) that locates and statistically tests effects in time-series data. It does so by using crossvalidation to identify time points to test, and then using a linear mixed effects model to actually perform the statistical test. More specifically, the data is subdivided in a number of subsets (by default 4). It takes one of the subsets (the *test* set) out of the full dataset, and conducts a linear mixed effects model on each sample of the remaining data (the *training* set). The sample with the highest absolute z value in the training set is used as the sample-to-be-tested for the test set. This procedure is repeated for all subsets of the data, and for all fixed effects in the model. Finally, a single linear mixed effects model is conducted for each fixed effects on the samples that were thus identified.

This packages also provides a function (`plot()`) to visualize time-series data to visually annotate the results of `find()`.

For a more detailed description, see the manuscript above.


## Installation

```
pip install time_series_test
```

## Dependencies

- [Python 3](https://www.python.org/)
- [datamatrix](https://pydatamatrix.eu/)
- [statsmodels](https://www.statsmodels.org/)
- [matplotlib](https://matplotlib.org/)


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

The model summary shows that, assuming an alpha level of .05, there are significant main effects of color type (z = -2.136, p = .033), set size (z = 17.2, p < .001), and a significant color-type by set-size interaction (z = 2.47, p = .014). However, we have selectively analyzed a sample range that we knew, based on a visual inspection of the data, to show these effects. This means that our analysis is circular: we have looked at the data to decide where to look! The `find()` function improves this by splitting the data into training and tests sets, as described under [About](#about), thus breaking the circularity.

```python
results = tst.find(dm, 'pupil ~ set_size * color_type',
                   groups='subject_nr', winlen=5)
```

The return value of `find()` is a `dict`, where keys are effect labels and values are named tuples of the following:

- `model`: a model as returned by `mixedlm().fit()`
- `samples`: a `set` with the sample indices that were used
- `p`: the p-value from the model
- `z`: the z-value from the model

The `summarize()` function is a convenient way to get the results in a human-readable format.

```python
print(tst.summarize(results))
```

We can pass the `results` to `plot()` to visualize the results:

```python
tst.plot(dm, dv='pupil', hue_factor='set_size', linestyle_factor='color_type',
         results=results)
plt.savefig('img/signal-plot-2.png')
```

![](https://github.com/smathot/time_series_test/raw/master/img/signal-plot-2.png)


## Function reference

[API]


## License

`time_series_test` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
