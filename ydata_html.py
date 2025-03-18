import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('./IMDB Dataset.csv/IMDB Dataset.csv')

profile = ProfileReport(
    df,
    title="Rapport d'analyse du dataset IMDB",
    explorative=True,
    minimal=False,
    samples=None,
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": False},
        "phi_k": {"calculate": False},
        "cramers": {"calculate": False},
    },
    vars={"num": {"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]}}
)

profile.to_file("rapport_imdb.html")

print("Le rapport sauvegardé 'rapport_imdb.html'")

