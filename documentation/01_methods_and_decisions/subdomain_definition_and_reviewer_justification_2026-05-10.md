# CPEC 2018 Subdomain Definition and Reviewer Justification

Date: 2026-05-10

## Purpose

This note documents why the project uses five CPEC analysis subdomains for transferability testing, area-of-applicability summaries, and reliability-aware exposure results. These labels are **analysis-domain labels**, not replacements for official administrative boundaries.

## Final Public Subdomain Labels

1. Kashgar (Xinjiang, China)
2. Gilgit-Baltistan
3. KP-AJK
4. Balochistan
5. Punjab-Sindh lowland corridor

## Basis for Creating the Subdomains

The subdomains were created by spatially joining the 2018 V3 sample points to political/administrative boundary shapefiles, then aggregating very small or legacy units where independent modelling would be statistically weak. The purpose is to support leave-one-domain-out transferability testing and subdomain-level reporting.

The grouping is justified by three requirements:

- **Spatial validation requirement:** random validation is not enough for spatially structured landslide data; grouped/spatial validation is needed to test geographic transferability.
- **Sample-size requirement:** each held-out domain must contain enough landslide and non-landslide samples. Islamabad and former FATA are too small by themselves; AJK is also too small for stable independent testing.
- **Planning interpretation requirement:** the final regions should remain understandable for CPEC/KKH planning and reporting.

## Raw Administrative Sample Counts

| admin_unit | n | positives | negatives | positive_share |
| --- | --- | --- | --- | --- |
| Azad Kashmir | 83 | 44 | 39 | 0.530 |
| Balochistan | 992 | 516 | 476 | 0.520 |
| Federally Administered Tribal Ar | 21 | 13 | 8 | 0.619 |
| Gilgit-Baltistan | 721 | 427 | 294 | 0.592 |
| Islamabad | 1 | 0 | 1 | 0.000 |
| Kashgar (Xinjiang, China) | 846 | 323 | 523 | 0.382 |
| Khyber-Pakhtunkhwa | 247 | 110 | 137 | 0.445 |
| Punjab | 274 | 60 | 214 | 0.219 |
| Sindh | 131 | 15 | 116 | 0.115 |

## Final Analysis-Domain Sample Counts

| cpec_admin_transfer_domain | n | positives | negatives | positive_share |
| --- | --- | --- | --- | --- |
| Kashgar (Xinjiang, China) | 846 | 323 | 523 | 0.382 |
| Gilgit-Baltistan | 721 | 427 | 294 | 0.592 |
| KP-AJK | 351 | 167 | 184 | 0.476 |
| Balochistan | 992 | 516 | 476 | 0.520 |
| Punjab-Sindh lowland corridor | 406 | 75 | 331 | 0.185 |

## Specific Naming Decisions

- **KP-AJK:** Khyber Pakhtunkhwa and Azad Jammu and Kashmir are grouped only for transfer-testing stability and northern-western mountainous corridor interpretation. This does not imply they are the same administrative unit.
- **Former FATA:** legacy GADM polygons labelled FATA are assigned to KP because former FATA was merged with Khyber Pakhtunkhwa through Pakistan's 25th Constitutional Amendment in 2018. It is not displayed as a separate current public domain.
- **Punjab-Sindh lowland corridor:** Punjab, Islamabad, and Sindh are grouped for modelling stability. Islamabad is a separate federal territory, but in this sample table it had only one sample, so it cannot support an independent held-out transfer test.
- **Kashgar and Gilgit-Baltistan:** kept separate because they are central to CPEC/KKH cross-border and high-mountain interpretation and have enough samples for testing.
- **Balochistan:** kept separate because it is large, geomorphically distinct, and has enough samples.

## Literature and Official Support

- Roberts et al. (2017), Ecography, recommend appropriate cross-validation strategies for spatial, temporal, and hierarchical data rather than ordinary random validation: https://doi.org/10.1111/ecog.02881
- Meyer and Pebesma (2021), Methods in Ecology and Evolution, define area of applicability for spatial prediction and emphasize where cross-validation error is expected to transfer: https://doi.org/10.1111/2041-210X.13650
- Official KP government portal uses KP for Khyber Pakhtunkhwa: https://kp.gov.pk/
- Pakistan Code / Constitution source for the Twenty-fifth Amendment context: https://pakistancode.gov.pk/
- AJK official portal supports the AJK/AJ&K naming convention: https://ajk.gov.pk/

## How to Describe This in the Paper

Suggested wording:

> To evaluate spatial transferability, the study area was partitioned into five politically interpretable analysis domains derived from administrative boundaries. Small or legacy units were aggregated where independent leave-one-domain-out testing would be statistically unstable. Former FATA polygons were assigned to KP following the 2018 constitutional merger, while AJK was grouped with KP only for transfer-testing stability and northern mountainous corridor interpretation. These subdomains were used for model transferability and reliability summaries, not as replacements for official administrative units.
