
## Processing the Data

Once the three raw CSVs are in this folder, run:

```bash
python scripts/clean_data.py
```

This produces the three `processed_*.csv` files used by the baseline notebook.

## Dataset Details

- **Nodes (transactions):** 203,769
- **Edges (payment flows):** 234,355
- **Features per node:** 166 (94 local + 72 aggregated neighborhood features, anonymized)
- **Labels:** class 1 = illicit (~2%), class 2 = licit (~21%), unknown (~77%)
- **Time steps:** 49 discrete steps, roughly two weeks apart

## License

**CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)

- **Attribution required:** Cite Elliptic and the Weber et al. (2019) paper
- **NonCommercial:** Non-commercial academic use only
- **NoDerivatives:** Modified versions of the raw dataset cannot be redistributed

Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0/

## Citation

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. *KDD '19 Workshop on Anomaly Detection in Finance*. https://arxiv.org/abs/1908.02591