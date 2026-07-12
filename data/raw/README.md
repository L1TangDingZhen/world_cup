# Raw Data

`international_results.csv` is downloaded by:

```bash
worldcup-predictor fetch-data
```

Source: <https://github.com/martj42/international_results>

License: CC0-1.0

The generated CSV and metadata file are intentionally ignored by Git. The metadata records the source URL and content hash needed to identify the exact local snapshot.

Important modeling limitation: the source describes scores as full-time including extra time, while the independent Poisson model is intended to model regulation-time goals. Knockout matches that went to extra time must be identified or excluded before production fitting.

