# EDA summary -- P_plus (2880 rows, 7 vars, alpha=0.05)

Binary (PRBS-type) columns, expected to fail normality regardless of transform: v1, v2, d

## Normality rejections (of 7 variables)

| Test | Before | After |
|---|---|---|
| shapiro wilk | 3 | 7 |
| dagostino k2 | 3 | 7 |
| jarque bera | 3 | 7 |
| anderson darling | 3 | 3 |
| **practically Gaussian** (\|skew\|<0.5, \|excess kurtosis\|<1.0) | 4 | 4 |

At n in the thousands, formal normality tests reject on departures too small to matter -- the effect-size row above is the one that actually answers whether the copula transform worked.

## Outliers

- Modified-z (MAD) outliers: 5
- IQR outliers: 87

## Hotelling's T² (post-copula-transform)

- UCL (alpha=0.05): 14.12
- Fraction of 2880 observations exceeding UCL: 0.025
  (expected ~0.05 under the null that the transform succeeded)