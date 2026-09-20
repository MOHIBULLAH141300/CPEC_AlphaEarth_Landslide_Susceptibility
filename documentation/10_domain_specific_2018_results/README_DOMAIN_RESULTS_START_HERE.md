# CPEC 2018 V3 Domain-Specific Results

These folders package the finalized 2018 V3 workflow by CPEC transfer domain.
Each domain contains raw stacked probability maps, transfer confidence, AOA,
reliability-aware products, leave-one-domain-out metrics, road exposure tables,
domain-wise TreeSHAP/factor-importance outputs, VIF diagnostics, and
publication-ready figures.

Important: the domain probability maps are clipped from the same validated
CPEC-wide 2018 model. They are not separately trained regional models. This is
intentional because the paper tests spatial transferability and feature-set
added value across subdomains.

Domains:
- `kashgar_xinjiang_china`: Kashgar (Xinjiang, China)
- `gilgit_baltistan`: Gilgit-Baltistan
- `kp_ajk`: KP-AJK
- `balochistan`: Balochistan
- `punjab_sindh_lowland_corridor`: Punjab-Sindh lowland corridor

Start with:

- `00_domain_comparison_summary\domainwise_mechanism_and_transferability_interpretation_report.md`
- `00_domain_comparison_summary\domain_transferability_mechanism_comparison_table.csv`
- `00_domain_comparison_summary\figure_domain_top_factor_shap_heatmap_fused.png`
- `00_domain_comparison_summary\figure_domain_transferability_reliability_comparison_heatmap.png`
